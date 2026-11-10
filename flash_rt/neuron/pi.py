"""π0 and π0.5 on AWS Neuron through JAX/PJRT.

The full SigLIP, Gemma prefix, and action expert execute on the selected
Neuron device. Stacked layer parameters and XLA loops bound compilation size.
Model math follows FlashRT's eager reference and the vendored OpenPI model.
Inputs to ``sample`` are normalized; token sequences contain no padding.
"""

from functools import partial
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

PAL = 'paligemma_with_expert.paligemma'
VP = PAL + '.model.vision_tower.vision_model'
EP = PAL + '.model.language_model.layers'
DP = 'paligemma_with_expert.gemma_expert.model.layers'
DN = 'paligemma_with_expert.gemma_expert.model.norm'


def linear(x, w, name):
    weight = w[name + '.weight']
    y = jnp.matmul(x.astype(weight.dtype), weight.T)
    if name + '.bias' in w:
        y = y + w[name + '.bias']
    return y


def rms(x, weight):
    z = x.astype(jnp.float32)
    z = z * jax.lax.rsqrt(jnp.mean(z*z, axis=-1, keepdims=True) + 1e-6)
    return (z * (1 + weight.astype(jnp.float32))).astype(x.dtype)


def layer_norm(x, w, name):
    z = x.astype(jnp.float32)
    mean = jnp.mean(z, axis=-1, keepdims=True)
    var = jnp.mean((z-mean)**2, axis=-1, keepdims=True)
    z = (z-mean) * jax.lax.rsqrt(var+1e-6)
    return (z*w[name+'.weight'].astype(jnp.float32) + w[name+'.bias'].astype(jnp.float32)).astype(x.dtype)


def ada_rms(x, w, name, cond):
    mod = linear(cond.astype(jnp.float32), w, name+'.dense').astype(jnp.float32)
    scale, shift, gate = jnp.split(mod, 3, axis=-1)
    z = x.astype(jnp.float32)
    z = z * jax.lax.rsqrt(jnp.mean(z*z, axis=-1, keepdims=True)+1e-6)
    return (z*(1+scale)+shift).astype(x.dtype), gate.astype(x.dtype)


def rope(x, pos, head_dim=256):
    shape = x.shape
    z = x.astype(jnp.float32).reshape(shape[0], -1, head_dim)
    inv = 10000.0 ** (-jnp.arange(head_dim//2, dtype=jnp.float32)/(head_dim//2))
    angle = pos[:, None] * inv[None, :]
    angle = jnp.concatenate([angle, angle], axis=-1)[:, None, :]
    rot = jnp.concatenate([-z[..., head_dim//2:], z[..., :head_dim//2]], axis=-1)
    return (z*jnp.cos(angle)+rot*jnp.sin(angle)).reshape(shape).astype(x.dtype)


def attention(q, k, v, mask=None):
    # [..., heads, sequence, head_dim]; singleton KV heads broadcast for GQA.
    logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)).astype(jnp.float32) * (q.shape[-1]**-0.5)
    if mask is not None:
        logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
    probs = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
    return jnp.matmul(probs, v)


def vision(images, w):
    nv = images.shape[0]
    patches = images.astype(jnp.float32).reshape(nv, 3, 16, 14, 16, 14)
    patches = patches.transpose(0, 2, 4, 1, 3, 5).reshape(nv, 256, 588)
    x = jnp.matmul(patches, w['patch.weight'].reshape(1152, -1).T) + w['patch.bias']
    x = (x + w['position']).astype(w['layers']['self_attn.q_proj.weight'].dtype)

    def block(x, layer):
        z = layer_norm(x, layer, 'layer_norm1')
        q, k, v = [linear(z, layer, 'self_attn.'+p+'_proj').reshape(nv, 256, 16, 72).transpose(0, 2, 1, 3) for p in ('q', 'k', 'v')]
        o = attention(q, k, v).transpose(0, 2, 1, 3).reshape(nv, 256, 1152)
        x = x + linear(o, layer, 'self_attn.out_proj')
        z = layer_norm(x, layer, 'layer_norm2')
        z = linear(jax.nn.gelu(linear(z, layer, 'mlp.fc1'), approximate=True), layer, 'mlp.fc2')
        return x+z, None

    x, _ = jax.lax.scan(block, x, w['layers'])
    x = layer_norm(x, w, 'post_layernorm')
    x = linear(x, w, 'projector')
    return x.reshape(nv*256, 2048)


def prefix(emb, layers):
    seq = emb.shape[0]
    pos = jnp.arange(seq, dtype=jnp.float32)

    def block(x, w):
        z = rms(x, w['input_layernorm.weight'])
        q, k, v = [linear(z, w, 'self_attn.'+p+'_proj') for p in ('q', 'k', 'v')]
        q, k = rope(q, pos), rope(k, pos)
        o = attention(q.reshape(seq, 8, 256).transpose(1, 0, 2), k[None], v[None])
        o = o.transpose(1, 0, 2).reshape(seq, 2048)
        x = x + linear(o, w, 'self_attn.o_proj')
        z = rms(x, w['post_attention_layernorm.weight'])
        z = jax.nn.gelu(linear(z, w, 'mlp.gate_proj'), approximate=True)*linear(z, w, 'mlp.up_proj')
        return x+linear(z, w, 'mlp.down_proj'), (k, v)

    _, cache = jax.lax.scan(block, emb, layers)
    return cache


def expert(x, cache, cond, w, pi05):
    seq = x.shape[0]
    plen = cache[0].shape[1]
    pos = jnp.arange(plen, plen+seq, dtype=jnp.float32)
    mask = jnp.ones((seq, plen+seq), dtype=jnp.bool_)
    if not pi05:
        # State forms its own prefix block, preceding the bidirectional action block.
        mask = mask.at[0, plen+1:].set(False)

    def norm(x, layer, name):
        if pi05:
            return ada_rms(x, layer, name, cond)
        return rms(x, layer[name+'.weight']), jnp.ones((), dtype=x.dtype)

    def block(x, args):
        layer, pk, pv = args
        z, gate = norm(x, layer, 'input_layernorm')
        q, k, v = [linear(z, layer, 'self_attn.'+p+'_proj') for p in ('q', 'k', 'v')]
        q, k = rope(q, pos), rope(k, pos)
        k, v = jnp.concatenate([pk, k]), jnp.concatenate([pv, v])
        o = attention(q.reshape(seq, 8, 256).transpose(1, 0, 2), k[None], v[None], mask[None])
        o = o.transpose(1, 0, 2).reshape(seq, 2048)
        x = x+linear(o, layer, 'self_attn.o_proj')*gate
        z, gate = norm(x, layer, 'post_attention_layernorm')
        z = jax.nn.gelu(linear(z, layer, 'mlp.gate_proj'), approximate=True)*linear(z, layer, 'mlp.up_proj')
        return x+linear(z, layer, 'mlp.down_proj')*gate, None

    x, _ = jax.lax.scan(block, x, (w['layers'], cache[0], cache[1]))
    return norm(x, w, 'norm')[0]


@partial(jax.jit, static_argnames=('pi05',))
def sample(w, images, tokens, state, noise, time_embeddings, *, pi05):
    vis = vision(images, w['vision'])
    lang = (w['embedding'][tokens] * math.sqrt(2048)).astype(vis.dtype)
    cache = prefix(jnp.concatenate([vis, lang]), w['encoder'])
    dtype = w['decoder']['layers']['self_attn.q_proj.weight'].dtype
    steps = time_embeddings.shape[0]

    def step(x, te):
        act = linear(x, w['edges'], 'action_in_proj')
        if pi05:
            cond = jax.nn.silu(linear(te, w['edges'], 'time_mlp_in'))
            cond = jax.nn.silu(linear(cond, w['edges'], 'time_mlp_out'))
        else:
            cond = jnp.zeros((1024,), dtype=jnp.float32)
            te = jnp.broadcast_to(te, act.shape)
            act = linear(jax.nn.silu(linear(jnp.concatenate([act, te], axis=-1), w['edges'], 'action_time_mlp_in')), w['edges'], 'action_time_mlp_out')
            act = jnp.concatenate([linear(state[None], w['edges'], 'state_proj'), act])
        out = expert(act.astype(dtype), cache, cond, w['decoder'], pi05)
        if not pi05:
            out = out[1:]
        velocity = linear(out, w['edges'], 'action_out_proj').astype(jnp.float32)
        return x - velocity/steps, None

    out, _ = jax.lax.scan(step, noise.astype(jnp.float32), time_embeddings)
    return out


def time_table(num_steps):
    if num_steps < 1:
        raise ValueError('num_steps must be positive')
    period = 4e-3 * (1000.0 ** np.linspace(0, 1, 512, dtype=np.float64))
    angles = (1 - np.arange(num_steps, dtype=np.float64)/num_steps)[:, None] * (2*np.pi/period)[None, :]
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1).astype(np.float32)


def load_weights(checkpoint, *, dtype=jnp.bfloat16, device=None):
    """Read a published OpenPI/LeRobot safetensors checkpoint on the CPU."""
    import torch
    from safetensors import safe_open
    path = Path(checkpoint)/'model.safetensors'
    with safe_open(path, framework='pt', device='cpu') as f:
        raw = {k.removeprefix('model.'): f.get_tensor(k).float().numpy() for k in f.keys()}

    def group(stem):
        return {k[len(stem)+1:]: v for k, v in raw.items() if k.startswith(stem+'.')}

    def layers(stem, count):
        first = group(stem+'.0')
        return {k: np.stack([raw[f'{stem}.{i}.{k}'] for i in range(count)]) for k in first}

    vw = {'layers': layers(VP+'.encoder.layers', 27),
          'patch.weight': raw[VP+'.embeddings.patch_embedding.weight'],
          'patch.bias': raw[VP+'.embeddings.patch_embedding.bias'],
          'position': raw[VP+'.embeddings.position_embedding.weight']}
    vw.update({'post_layernorm.'+k: v for k, v in group(VP+'.post_layernorm').items()})
    vw.update({'projector.'+k: v for k, v in group(PAL+'.model.multi_modal_projector.linear').items()})
    dw = {'layers': layers(DP, 18)}
    dw.update({'norm.'+k: v for k, v in group(DN).items()})
    emb_key = PAL+'.lm_head.weight'
    if emb_key not in raw:
        emb_key = PAL+'.model.language_model.embed_tokens.weight'
    weights = {'vision': vw, 'encoder': layers(EP, 18), 'decoder': dw,
               'embedding': raw[emb_key],
               'edges': {k: v for k, v in raw.items() if k.startswith(('action_', 'state_proj.', 'time_mlp_'))}}

    def put(path, value):
        names = [str(p.key) for p in path]
        fp32 = names[0] == 'edges' or any(s in '.'.join(names) for s in ('norm', 'patch.'))
        array = jnp.asarray(value, dtype=jnp.float32 if fp32 else dtype)
        return jax.device_put(array, device) if device is not None else array

    # Avoid implicitly initializing or loading weights into the wrong accelerator.
    with jax.default_device(jax.devices('cpu')[0]):
        return jax.tree_util.tree_map_with_path(put, weights)


class PiNeuronFrontend:
    """FlashRT frontend for batch-one π0/π0.5 inference on Neuron.

    ``infer_normalized`` exposes the exact tensor boundary for verification.
    ``infer`` handles uint8 images, state normalization and robot action scaling.
    """
    def __init__(self, checkpoint_dir, *, config='pi05', num_views=2,
                 num_steps=10, action_horizon=None, device=None):
        if config not in ('pi0', 'pi05'):
            raise ValueError('Neuron Pi backend supports pi0 and pi05')
        self.pi05 = config == 'pi05'
        self.num_views = num_views
        ckpt = Path(checkpoint_dir)
        cfg = json.loads((ckpt/'config.json').read_text()) if (ckpt/'config.json').exists() else {}
        self.chunk_size = action_horizon or cfg.get('chunk_size', 50)
        self.device = device or next((d for d in jax.devices() if d.platform == 'neuron'), None)
        if self.device is None or self.device.platform != 'neuron':
            raise RuntimeError('A Neuron PJRT device is required; CPU fallback is disabled')
        self.weights = load_weights(ckpt, device=self.device)
        self.times = jax.device_put(time_table(num_steps), self.device)
        from flash_rt.core.utils.norm_stats import load_norm_stats
        self.norm_stats = load_norm_stats([ckpt/'norm_stats.json', ckpt/'assets/physical-intelligence/libero/norm_stats.json'], checkpoint_dir=ckpt, strict=False)
        if self.norm_stats is None:
            from safetensors import safe_open
            with safe_open(ckpt/'model.safetensors', framework='pt', device='cpu') as f:
                self.norm_stats = {
                    'state': {s: f.get_tensor('normalize_inputs.buffer_observation_state.'+s).float().numpy() for s in ('mean', 'std')},
                    'actions': {s: f.get_tensor('unnormalize_outputs.buffer_action.'+s).float().numpy() for s in ('mean', 'std')},
                }
        self.tokens = None
        self.rng = np.random.default_rng()

    def set_prompt(self, prompt):
        from flash_rt.utils.paligemma_tokenizer import load_paligemma_sentencepiece, encode_pi05_prompt
        sp = load_paligemma_sentencepiece()
        self.tokens = np.asarray(encode_pi05_prompt(sp, prompt), dtype=np.int32)

    def infer_normalized(self, images, tokens, state, noise):
        arrays = [np.asarray(images, dtype=np.float32), np.asarray(tokens, dtype=np.int32),
                  np.asarray(state, dtype=np.float32), np.asarray(noise, dtype=np.float32)]
        if arrays[0].shape != (self.num_views, 3, 224, 224) or arrays[2].shape != (32,) or arrays[3].shape != (self.chunk_size, 32):
            raise ValueError('Expected images [views,3,224,224], state [32], noise [horizon,32]')
        if arrays[1].ndim != 1 or arrays[1].size == 0:
            raise ValueError('Tokens must be a nonempty sequence without padding')
        args = [jax.device_put(x, self.device) for x in arrays]
        return sample(self.weights, *args, self.times, pi05=self.pi05)

    def infer(self, obs):
        if self.tokens is None:
            raise ValueError('Call set_prompt before infer')
        images = obs.get('images')
        if images is None:
            images = [obs[k] for k in ('image', 'wrist_image', 'wrist_image_right') if k in obs]
        if len(images) != self.num_views:
            raise ValueError('Image count differs from configured num_views')
        if any(np.asarray(x).shape != (224, 224, 3) or np.asarray(x).dtype != np.uint8 for x in images):
            raise ValueError('Public inference expects uint8 HWC images at 224x224')
        pixels = np.stack(images).transpose(0, 3, 1, 2).astype(np.float32)/127.5-1
        state = np.zeros(32, dtype=np.float32)
        if not self.pi05:
            if 'state' not in obs:
                raise ValueError('π0 requires robot state')
            values = np.asarray(obs['state'], dtype=np.float32).reshape(-1)
            if len(values) > 32:
                raise ValueError('State exceeds model width')
            stats = self.norm_stats['state']
            n = len(values)
            if 'q01' in stats:
                state[:n] = (values-np.asarray(stats['q01'])[:n])/(np.asarray(stats['q99'])[:n]-np.asarray(stats['q01'])[:n]+1e-6)*2-1
            else:
                state[:n] = (values-np.asarray(stats['mean'])[:n])/(np.asarray(stats['std'])[:n]+1e-8)
        noise = obs.get('noise', self.rng.standard_normal((self.chunk_size, 32)).astype(np.float32))
        raw = np.asarray(self.infer_normalized(pixels, self.tokens, state, noise))
        from flash_rt.core.utils.actions import unnormalize_actions
        stats = self.norm_stats['actions']
        if 'q01' in stats:
            actions = unnormalize_actions(raw, self.norm_stats)
            width = len(stats['q01'])
        else:
            width = len(stats['mean'])
            actions = raw[:, :width]*np.asarray(stats['std'])+np.asarray(stats['mean'])
        return {'actions': actions[:, :width], 'normalized_actions': raw}
