"""Cora feature methods. Register with data.registry on import."""
from __future__ import annotations

from my_code.data.registry import register_feature
from my_code.data.cora.feature.add_degree import build_feature_cache as build_add_degree, load_feature_cache as load_add_degree
from my_code.data.cora.feature.main import build_feature_cache as build_main, load_feature_cache as load_main

register_feature("cora", "add_degree", build_add_degree, load_add_degree)
register_feature("cora", "main", build_main, load_main)
