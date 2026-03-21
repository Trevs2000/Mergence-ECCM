import numpy as np
from typing import Dict
from .psc import PSCCalculator
from .fsc import FSCCalculator
from .rsc import RSCCalculator
from .epc import EPCTrainer

class ECCMCalculator:
    """
    Evolutionary Context Compatibility Measure
    Combines all metrics with learned weights.
    
    ECCM = w_psc * PSC + w_fsc * FSC + w_rsc * RSC + w_epc * EPC_pred
    
    Weights are learned from merge history (via EPC).
    """
    
    def __init__(self,  w_psc: float = 0.156, w_fsc: float = 0.743,
                 w_rsc: float = 0.101, w_epc: float = 0.0):
        """
        Default: equal weights (will optimize later)
        """
        self.w_psc = w_psc
        self.w_fsc = w_fsc
        self.w_rsc = w_rsc
        self.w_epc = w_epc
        
        self.psc_calc = PSCCalculator(method='cosine')
        self.fsc_calc = FSCCalculator(strategy='correlation')
        self.rsc_calc = RSCCalculator()
        self.epc_trainer = EPCTrainer()
    
    def compute(self, model_a, model_b, X: np.ndarray, 
                epc_pred: float = 0.0) -> Dict[str, float]:
        """
        Compute ECCM score and component breakdown.
        
        Args:
            model_a, model_b: models to merge
            X: validation data
            epc_pred: Pre-computed EPC prediction
        
        Returns:
            dict with all component scores
        """
        # Compute each metric
        psc = self.psc_calc.compute(model_a, model_b)
        fsc = self.fsc_calc.compute(model_a, model_b, X)
        rsc = self.rsc_calc.compute(model_a, model_b, X)
        
        # Compute ECCM
        eccm = (self.w_psc * psc + 
                self.w_fsc * fsc + 
                self.w_rsc * rsc + 
                self.w_epc * epc_pred)
        
        return {
            'psc': float(psc),
            'fsc': float(fsc),
            'rsc': float(rsc),
            'epc': float(epc_pred),
            'eccm': float(eccm)
        }