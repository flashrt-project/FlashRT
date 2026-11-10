# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored + patched nkilib TKG attention for head_dim=256. See
# attention_tkg.py's banner for provenance and the exact head_dim-on-partition
# patch sites. Unchanged AWS helpers are imported from the installed nkilib.

from .attention_tkg import attention_tkg
from .attention_tkg_utils import AttnTKGConfig

__all__ = ["attention_tkg", "AttnTKGConfig"]
