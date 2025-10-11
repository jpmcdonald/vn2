"""SIP/SLURP: Stochastic Information Packets and Libraries with Relationships Preserved"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


# ---- SIP helpers ----

def make_sip_from_uniform_threshold(
    u: np.ndarray, 
    threshold: float, 
    value_if_success: float, 
    value_if_fail: float = 0.0
) -> np.ndarray:
    """
    Create SIP from uniform draws with threshold logic.
    Mirrors R example: if u > threshold -> fail_value, else success_value
    
    Args:
        u: Uniform(0,1) draws
        threshold: Success threshold
        value_if_success: Value when u <= threshold
        value_if_fail: Value when u > threshold
        
    Returns:
        SIP array
    """
    return np.where(u > threshold, value_if_fail, value_if_success)


# ---- SLURP: library of SIPs with relationships preserved ----

@dataclass
class SLURP:
    """
    Stochastic Library Unit with Relationships Preserved.
    
    A collection of SIPs where each row represents a scenario,
    preserving correlations/dependencies across variables.
    """
    data: pd.DataFrame                  # rows = scenarios, columns = SIP names
    meta: Optional[pd.DataFrame] = None # index = meta fields, columns = SIP names
    provenance: str = ""
    
    @staticmethod
    def from_dict(
        sips: Dict[str, Sequence[float]], 
        meta: Optional[pd.DataFrame] = None, 
        provenance: str = ""
    ) -> "SLURP":
        """
        Create SLURP from dictionary of SIP arrays.
        
        Args:
            sips: Dict mapping SIP names to value arrays (all same length)
            meta: Metadata DataFrame (rows=fields, cols=SIP names)
            provenance: Description of data source
            
        Returns:
            SLURP object
        """
        lengths = {len(v) for v in sips.values()}
        if len(lengths) != 1:
            raise ValueError("All SIP arrays must have the same length")
        
        df = pd.DataFrame(sips)
        
        if meta is not None and not set(df.columns).issubset(set(meta.columns)):
            raise ValueError("Meta must have columns for all SIP names")
        
        return SLURP(df, meta, provenance)
    
    @property
    def n_scenarios(self) -> int:
        """Number of scenarios (rows)"""
        return len(self.data)
    
    @property
    def names(self) -> Iterable[str]:
        """SIP names (columns)"""
        return self.data.columns
    
    def sample_rows(
        self, 
        n: int, 
        replace: bool = True, 
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Sample full scenarios (rows) to preserve relationships.
        
        Args:
            n: Number of scenarios to sample
            replace: Whether to sample with replacement
            seed: Random seed
            
        Returns:
            DataFrame of sampled scenarios
        """
        rng = np.random.default_rng(seed)
        
        if replace:
            idx = rng.integers(0, self.n_scenarios, size=n)
        else:
            if n > self.n_scenarios:
                raise ValueError(f"Cannot sample {n} without replacement from {self.n_scenarios}")
            idx = rng.choice(self.n_scenarios, size=n, replace=False)
        
        return self.data.iloc[idx].reset_index(drop=True)
    
    def to_xml(
        self, 
        path: str, 
        csvr: int = 4, 
        average: bool = False, 
        median: bool = False
    ) -> None:
        """
        Export SLURP to XML format (SIPmath compatible).
        
        Args:
            path: Output file path
            csvr: Number of decimal places to round
            average: Include average attribute
            median: Include median attribute
        """
        root = ET.Element("SLURP", {
            "name": "SLURP",
            "provenance": self.provenance,
            "count": str(self.data.shape[1]),
        })
        
        for col in self.data.columns:
            attrs = {
                "name": col,
                "count": str(self.data.shape[0]),
                "type": "CSV",
            }
            
            if average:
                attrs["average"] = f"{self.data[col].mean():.{csvr}f}"
            if median:
                attrs["median"] = f"{self.data[col].median():.{csvr}f}"
            
            # Attach metadata fields as attributes if provided
            if self.meta is not None and col in self.meta.columns:
                for meta_field, meta_value in self.meta[col].items():
                    attrs[str(meta_field)] = str(meta_value)
            
            sip_el = ET.SubElement(root, "SIP", attrs)
            
            # CSV payload
            vals = np.round(self.data[col].to_numpy(dtype=float), csvr)
            sip_el.text = ", ".join(f"{x:.{csvr}f}" for x in vals)
        
        tree = ET.ElementTree(root)
        
        # Pretty print if available (Python 3.9+)
        try:
            ET.indent(tree, space="  ", level=0)
        except AttributeError:
            pass
        
        tree.write(path, encoding="utf-8", xml_declaration=True)


# ---- Metadata builder (R SIPMetaDF equivalent) ----

def sip_meta_df(
    df: pd.DataFrame, 
    idvect: Sequence, 
    metanamesvect: Sequence[str]
) -> pd.DataFrame:
    """
    Build metadata DataFrame from original data.
    
    Args:
        df: Original DataFrame (first column contains IDs)
        idvect: List of IDs to include (SIP names)
        metanamesvect: Column names to extract as metadata
        
    Returns:
        DataFrame with rows = metadata fields, columns = SIP names
    """
    id_col = df.columns[0]
    out = pd.DataFrame(index=metanamesvect, columns=idvect, dtype=str)
    
    loc = df.set_index(id_col)
    
    for sip_name in idvect:
        if sip_name not in loc.index:
            raise KeyError(f"ID {sip_name} not found in df[{id_col}]")
        
        for m in metanamesvect:
            out.at[m, sip_name] = str(loc.at[sip_name, m])
    
    return out

