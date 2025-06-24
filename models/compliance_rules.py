from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class ComplianceRule:
    """
    Represents a single compliance rule
    """
    rule_id: str
    rule_name: str
    description: Optional[str] = None
    severity: str = "medium"
    is_active: bool = True
    conditions: Dict = field(default_factory=dict)

@dataclass
class ComplianceRuleSet:
    """
    Collection of compliance rules for a specific service
    """
    service_id: str
    service_name: str
    rules: List[ComplianceRule] = field(default_factory=list)
    
    def get_rule_by_id(self, rule_id: str) -> Optional[ComplianceRule]:
        """
        Retrieve a specific rule by its ID
        
        Args:
            rule_id (str): Unique identifier for the rule
        
        Returns:
            ComplianceRule or None: Matching rule
        """
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None
    
    def validate_ruleset(self) -> bool:
        """
        Validate the integrity of the ruleset
        
        Returns:
            bool: Whether the ruleset is valid
        """
        # Check for duplicate rule IDs
        rule_ids = [rule.rule_id for rule in self.rules]
        return len(rule_ids) == len(set(rule_ids))

def load_compliance_rules_from_config(config_path: str) -> Optional[ComplianceRuleSet]:
    """
    Load compliance rules from a configuration file
    
    Args:
        config_path (str): Path to the configuration file
    
    Returns:
        ComplianceRuleSet or None: Loaded ruleset
    """
    try:
        import json
        
        with open(config_path, 'r') as config_file:
            config_data = json.load(config_file)
        
        # Create ComplianceRuleSet from config
        ruleset = ComplianceRuleSet(
            service_id=config_data.get('service_id', ''),
            service_name=config_data.get('service_name', ''),
            rules=[
                ComplianceRule(
                    rule_id=rule.get('rule_id', ''),
                    rule_name=rule.get('rule_name', ''),
                    description=rule.get('description', ''),
                    severity=rule.get('severity', 'medium'),
                    is_active=rule.get('is_active', True),
                    conditions=rule.get('conditions', {})
                )
                for rule in config_data.get('rules', [])
            ]
        )
        
        # Validate ruleset
        if ruleset.validate_ruleset():
            return ruleset
        
        return None
    
    except Exception as e:
        print(f"Error loading compliance rules: {e}")
        return None