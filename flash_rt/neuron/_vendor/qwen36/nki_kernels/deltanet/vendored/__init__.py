# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored nkilib qkv_tkg, extended with the caller-selectable i_column_tiling
# projection path. See qkv_tkg.py's banner for provenance and the exact
# divergence. Unchanged AWS helpers are imported from the installed nkilib.

from .qkv_tkg import qkv_tkg

__all__ = ["qkv_tkg"]
