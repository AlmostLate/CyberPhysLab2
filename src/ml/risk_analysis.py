"""
Risk analysis tools for credit scoring.

This module provides risk analysis functionality including
risk profiling, trend analysis, and portfolio risk assessment.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import MCPToolsConfig


class RiskAnalyzer:
    """
    Risk analyzer for credit scoring.
    
    This class provides methods for analyzing credit risk at both
    individual and portfolio levels.
    
    Example:
        >>> analyzer = RiskAnalyzer()
        >>> risk_profile = analyzer.analyze_individual(age=35, income=75000)
    """
    
    def __init__(self):
        """Initialize risk analyzer."""
        self.risk_levels = MCPToolsConfig.RISK_LEVELS
        self.age_groups = MCPToolsConfig.AGE_GROUPS
    
    def analyze_individual(
        self,
        age: int,
        income: float,
        employment_years: int,
        education_level: str,
        marital_status: str = "Unknown",
        has_credit_card: bool = False,
        has_mortgage: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze risk for an individual client.
        
        Args:
            age: Client's age
            income: Annual income
            employment_years: Years of employment
            education_level: Education level
            marital_status: Marital status
            has_credit_card: Has credit card
            has_mortgage: Has mortgage
        
        Returns:
            Dict with risk analysis results
        """
        risk_factors = []
        risk_score = 0
        
        # Age risk
        age_risk = self._analyze_age_risk(age)
        risk_factors.append({"factor": "Age", "value": age, "risk": age_risk})
        risk_score += age_risk["score"]
        
        # Income risk
        income_risk = self._analyze_income_risk(income)
        risk_factors.append({"factor": "Income", "value": income, "risk": income_risk})
        risk_score += income_risk["score"]
        
        # Employment risk
        employment_risk = self._analyze_employment_risk(employment_years)
        risk_factors.append({"factor": "Employment", "value": employment_years, "risk": employment_risk})
        risk_score += employment_risk["score"]
        
        # Education risk
        education_risk = self._analyze_education_risk(education_level)
        risk_factors.append({"factor": "Education", "value": education_level, "risk": education_risk})
        risk_score += education_risk["score"]
        
        # Credit history
        credit_risk = self._analyze_credit_history_risk(has_credit_card, has_mortgage)
        risk_factors.append({"factor": "Credit History", "value": "", "risk": credit_risk})
        risk_score += credit_risk["score"]
        
        # Determine overall risk level
        risk_level = self._determine_risk_level(risk_score)
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "recommendation": self._get_recommendation(risk_level),
            "max_risk_score": 100
        }
    
    def _analyze_age_risk(self, age: int) -> Dict[str, Any]:
        """Analyze risk based on age."""
        if 30 <= age <= 50:
            return {"level": "low", "score": 0, "description": "Prime working age"}
        elif 18 <= age < 30:
            return {"level": "medium", "score": 20, "description": "Young, limited credit history"}
        elif 50 < age <= 65:
            return {"level": "low", "score": 5, "description": "Mature, stable"}
        else:
            return {"level": "high", "score": 30, "description": "Non-standard age"}
    
    def _analyze_income_risk(self, income: float) -> Dict[str, Any]:
        """Analyze risk based on income."""
        if income >= 100000:
            return {"level": "low", "score": 0, "description": "High income"}
        elif income >= 60000:
            return {"level": "low", "score": 10, "description": "Above average income"}
        elif income >= 40000:
            return {"level": "medium", "score": 25, "description": "Average income"}
        else:
            return {"level": "high", "score": 40, "description": "Below average income"}
    
    def _analyze_employment_risk(self, years: int) -> Dict[str, Any]:
        """Analyze risk based on employment history."""
        if years >= 10:
            return {"level": "low", "score": 0, "description": "Stable employment"}
        elif years >= 5:
            return {"level": "low", "score": 10, "description": "Moderate stability"}
        elif years >= 2:
            return {"level": "medium", "score": 25, "description": "Some instability"}
        else:
            return {"level": "high", "score": 40, "description": "Limited employment history"}
    
    def _analyze_education_risk(self, education: str) -> Dict[str, Any]:
        """Analyze risk based on education level."""
        education_risks = {
            "Doctorate": {"level": "low", "score": 0},
            "Master's": {"level": "low", "score": 5},
            "Bachelor's": {"level": "low", "score": 10},
            "HS-grad": {"level": "medium", "score": 20},
            "Some-college": {"level": "medium", "score": 25},
            "Other": {"level": "high", "score": 30}
        }
        
        result = education_risks.get(education, {"level": "medium", "score": 20})
        result["description"] = f"Education: {education}"
        return result
    
    def _analyze_credit_history_risk(
        self,
        has_credit_card: bool,
        has_mortgage: bool
    ) -> Dict[str, Any]:
        """Analyze risk based on credit history."""
        score = 0
        factors = []
        
        if has_credit_card:
            score -= 10  # Reduces risk
            factors.append("Credit card")
        
        if has_mortgage:
            score -= 15  # Reduces risk
            factors.append("Mortgage")
        
        if score <= -20:
            level = "low"
            description = f"Excellent credit history ({', '.join(factors)})"
        elif score <= -5:
            level = "low"
            description = f"Good credit history ({', '.join(factors)})" if factors else "No negative factors"
        elif score == 0:
            level = "medium"
            description = "Limited credit history"
        else:
            level = "high"
            description = "No established credit"
        
        return {"level": level, "score": max(0, score + 20), "description": description}
    
    def _determine_risk_level(self, risk_score: int) -> str:
        """Determine risk level from score."""
        if risk_score <= 25:
            return "low"
        elif risk_score <= 50:
            return "medium"
        elif risk_score <= 75:
            return "high"
        else:
            return "very_high"
    
    def _get_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on risk level."""
        recommendations = {
            "low": "Approve with standard terms. Client shows excellent risk profile.",
            "medium": "Approve with modified terms. Consider higher interest rate or additional verification.",
            "high": "Require additional documentation and collateral. Higher interest rate required.",
            "very_high": "Decline or require co-signer and substantial collateral."
        }
        return recommendations.get(risk_level, "Review manually.")
    
    def analyze_portfolio(
        self,
        predictions: List[int],
        probabilities: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze risk at portfolio level.
        
        Args:
            predictions: List of predictions (0/1)
            probabilities: List of default probabilities
        
        Returns:
            Dict with portfolio risk analysis
        """
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        n = len(predictions)
        
        # Basic stats
        total_approved = (predictions == 1).sum()
        approval_rate = total_approved / n
        avg_probability = probabilities.mean()
        
        # Risk distribution
        low_risk = (probabilities < 0.2).sum()
        medium_risk = ((probabilities >= 0.2) & (probabilities < 0.5)).sum()
        high_risk = (probabilities >= 0.5).sum()
        
        # Expected loss
        expected_loss = probabilities.sum() / n
        
        return {
            "total_applications": n,
            "approved": int(total_approved),
            "rejected": int(n - total_approved),
            "approval_rate": float(approval_rate),
            "average_default_probability": float(avg_probability),
            "risk_distribution": {
                "low": int(low_risk),
                "medium": int(medium_risk),
                "high": int(high_risk)
            },
            "expected_loss_rate": float(expected_loss),
            "portfolio_health": "excellent" if avg_probability < 0.2 else "good" if avg_probability < 0.35 else "poor"
        }


def calculate_default_probability(
    income: float,
    age: int,
    employment_years: int,
    education_level: str
) -> float:
    """
    Calculate simple default probability estimate.
    
    Args:
        income: Annual income
        age: Client age
        employment_years: Years of employment
        education_level: Education level
    
    Returns:
        float: Estimated default probability (0 to 1)
    """
    prob = 0.5  # Base probability
    
    # Income factor (higher income = lower default risk)
    if income >= 100000:
        prob -= 0.25
    elif income >= 60000:
        prob -= 0.15
    elif income >= 40000:
        prob -= 0.05
    
    # Age factor
    if 30 <= age <= 50:
        prob -= 0.05
    elif age < 25:
        prob += 0.10
    
    # Employment factor
    if employment_years >= 10:
        prob -= 0.10
    elif employment_years < 2:
        prob += 0.10
    
    # Education factor
    if education_level in ["Doctorate", "Master's", "Bachelor's"]:
        prob -= 0.05
    
    # Clamp to valid range
    return max(0.01, min(0.99, prob))


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("Risk Analyzer module loaded")
    
    analyzer = RiskAnalyzer()
    
    # Test individual analysis
    print("\n1. Individual Risk Analysis:")
    result = analyzer.analyze_individual(
        age=35,
        income=75000,
        employment_years=10,
        education_level="Bachelor's"
    )
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Risk Score: {result['risk_score']}")
    print(f"   Recommendation: {result['recommendation']}")
    
    # Test default probability
    print("\n2. Default Probability Calculation:")
    prob = calculate_default_probability(
        income=75000,
        age=35,
        employment_years=10,
        education_level="Bachelor's"
    )
    print(f"   Default Probability: {prob:.2%}")
