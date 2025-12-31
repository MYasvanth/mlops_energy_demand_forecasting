"""
Automated Alerting System for Energy Demand Forecasting

This module implements configurable alerting thresholds and notification
system for monitoring energy demand forecasting models.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class AlertRule:
    """
    Represents a single alert rule with conditions and thresholds.
    """

    def __init__(self, name: str, metric: str, condition: str, threshold: float,
                 severity: str = 'warning', description: str = '', enabled: bool = True):
        """
        Initialize alert rule.

        Args:
            name (str): Rule name.
            metric (str): Metric to monitor.
            condition (str): Condition ('>', '<', '>=', '<=', '==', '!=').
            threshold (float): Threshold value.
            severity (str): Alert severity ('info', 'warning', 'error', 'critical').
            description (str): Rule description.
            enabled (bool): Whether rule is enabled.
        """
        self.name = name
        self.metric = metric
        self.condition = condition
        self.threshold = threshold
        self.severity = severity.lower()
        self.description = description
        self.enabled = enabled
        self.last_triggered = None
        self.trigger_count = 0

    def evaluate(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate the rule against current metrics.

        Args:
            metrics (Dict[str, Any]): Current metrics.

        Returns:
            Optional[Dict[str, Any]]: Alert details if triggered, None otherwise.
        """
        if not self.enabled:
            return None

        # Extract metric value (support nested metrics)
        metric_value = self._get_nested_value(metrics, self.metric)
        if metric_value is None:
            return None

        # Evaluate condition
        triggered = self._check_condition(metric_value, self.condition, self.threshold)

        if triggered:
            self.last_triggered = datetime.now()
            self.trigger_count += 1

            return {
                'rule_name': self.name,
                'metric': self.metric,
                'value': metric_value,
                'condition': f"{self.condition} {self.threshold}",
                'severity': self.severity,
                'description': self.description,
                'timestamp': self.last_triggered.isoformat(),
                'trigger_count': self.trigger_count
            }

        return None

    def _get_nested_value(self, data: Dict[str, Any], key_path: str) -> Any:
        """
        Get nested value from dictionary using dot notation.

        Args:
            data (Dict[str, Any]): Data dictionary.
            key_path (str): Key path (e.g., 'peak_demand.peak_accuracy').

        Returns:
            Any: Value at key path, None if not found.
        """
        keys = key_path.split('.')
        current = data

        try:
            for key in keys:
                if isinstance(current, dict):
                    current = current[key]
                else:
                    return None
            return current
        except (KeyError, TypeError):
            return None

    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """
        Check if condition is met.

        Args:
            value (float): Current value.
            condition (str): Condition operator.
            threshold (float): Threshold value.

        Returns:
            bool: True if condition met.
        """
        if condition == '>':
            return value > threshold
        elif condition == '<':
            return value < threshold
        elif condition == '>=':
            return value >= threshold
        elif condition == '<=':
            return value <= threshold
        elif condition == '==':
            return abs(value - threshold) < 1e-6  # Float comparison
        elif condition == '!=':
            return abs(value - threshold) >= 1e-6
        else:
            logger.warning(f"Unknown condition: {condition}")
            return False


class AlertManager:
    """
    Manages alert rules and notifications.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize alert manager.

        Args:
            config_path (Optional[str]): Path to configuration file.
        """
        self.rules: Dict[str, AlertRule] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.notification_channels: Dict[str, Callable] = {}
        self.config = self._load_config(config_path) if config_path else self._get_default_config()

        # Setup notification channels
        self._setup_email_channel()
        self._setup_slack_channel()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.

        Args:
            config_path (str): Configuration file path.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {str(e)}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.

        Returns:
            Dict[str, Any]: Default configuration.
        """
        return {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'from_email': '',
                'to_emails': []
            },
            'slack': {
                'enabled': False,
                'webhook_url': '',
                'channel': '#alerts'
            },
            'alert_rules': []
        }

    def add_rule(self, rule: AlertRule) -> None:
        """
        Add an alert rule.

        Args:
            rule (AlertRule): Alert rule to add.
        """
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> None:
        """
        Remove an alert rule.

        Args:
            rule_name (str): Name of rule to remove.
        """
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")

    def load_rules_from_config(self, rules_config: List[Dict[str, Any]]) -> None:
        """
        Load alert rules from configuration.

        Args:
            rules_config (List[Dict[str, Any]]): List of rule configurations.
        """
        for rule_config in rules_config:
            rule = AlertRule(
                name=rule_config['name'],
                metric=rule_config['metric'],
                condition=rule_config['condition'],
                threshold=rule_config['threshold'],
                severity=rule_config.get('severity', 'warning'),
                description=rule_config.get('description', ''),
                enabled=rule_config.get('enabled', True)
            )
            self.add_rule(rule)

    def evaluate_all_rules(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate all alert rules against current metrics.

        Args:
            metrics (Dict[str, Any]): Current metrics.

        Returns:
            List[Dict[str, Any]]: List of triggered alerts.
        """
        triggered_alerts = []

        for rule in self.rules.values():
            alert = rule.evaluate(metrics)
            if alert:
                triggered_alerts.append(alert)
                self.alert_history.append(alert)

        # Keep only recent alerts (last 1000)
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

        return triggered_alerts

    def send_notifications(self, alerts: List[Dict[str, Any]]) -> None:
        """
        Send notifications for triggered alerts.

        Args:
            alerts (List[Dict[str, Any]]): List of alerts to notify about.
        """
        if not alerts:
            return

        # Group alerts by severity
        alerts_by_severity = {}
        for alert in alerts:
            severity = alert['severity']
            if severity not in alerts_by_severity:
                alerts_by_severity[severity] = []
            alerts_by_severity[severity].append(alert)

        # Send notifications for each severity level
        for severity, severity_alerts in alerts_by_severity.items():
            self._send_email_notification(severity_alerts, severity)
            self._send_slack_notification(severity_alerts, severity)

    def _setup_email_channel(self) -> None:
        """Setup email notification channel."""
        self.notification_channels['email'] = self._send_email_notification

    def _setup_slack_channel(self) -> None:
        """Setup Slack notification channel."""
        self.notification_channels['slack'] = self._send_slack_notification

    def _send_email_notification(self, alerts: List[Dict[str, Any]], severity: str) -> None:
        """
        Send email notification.

        Args:
            alerts (List[Dict[str, Any]]): Alerts to send.
            severity (str): Alert severity.
        """
        email_config = self.config.get('email', {})
        if not email_config.get('enabled', False):
            return

        try:
            # Create message
            subject = f"Energy Demand Forecasting Alert - {severity.upper()}"
            body = self._format_alert_message(alerts, 'email')

            # Create email
            msg = MIMEMultipart()
            msg['From'] = email_config['from_email']
            msg['To'] = ', '.join(email_config['to_emails'])
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'html'))

            # Send email
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            text = msg.as_string()
            server.sendmail(email_config['from_email'], email_config['to_emails'], text)
            server.quit()

            logger.info(f"Email notification sent for {len(alerts)} {severity} alerts")

        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}")

    def _send_slack_notification(self, alerts: List[Dict[str, Any]], severity: str) -> None:
        """
        Send Slack notification.

        Args:
            alerts (List[Dict[str, Any]]): Alerts to send.
            severity (str): Alert severity.
        """
        slack_config = self.config.get('slack', {})
        if not slack_config.get('enabled', False):
            return

        try:
            webhook_url = slack_config['webhook_url']
            body = self._format_alert_message(alerts, 'slack')

            payload = {
                'channel': slack_config.get('channel', '#alerts'),
                'text': f"🚨 Energy Demand Forecasting Alert - {severity.upper()}",
                'attachments': [{
                    'text': body,
                    'color': self._get_severity_color(severity)
                }]
            }

            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()

            logger.info(f"Slack notification sent for {len(alerts)} {severity} alerts")

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {str(e)}")

    def _format_alert_message(self, alerts: List[Dict[str, Any]], format_type: str) -> str:
        """
        Format alert message for notifications.

        Args:
            alerts (List[Dict[str, Any]]): Alerts to format.
            format_type (str): Format type ('email' or 'slack').

        Returns:
            str: Formatted message.
        """
        if format_type == 'email':
            message = "<h2>Energy Demand Forecasting Alerts</h2><ul>"
            for alert in alerts:
                message += f"""
                <li>
                    <strong>{alert['rule_name']}</strong><br>
                    Metric: {alert['metric']}<br>
                    Value: {alert['value']:.4f}<br>
                    Condition: {alert['condition']}<br>
                    Severity: {alert['severity']}<br>
                    {alert['description']}<br>
                    Time: {alert['timestamp']}
                </li>
                """
            message += "</ul>"
        else:  # Slack format
            message = ""
            for alert in alerts:
                message += f"• *{alert['rule_name']}*\n"
                message += f"  Metric: {alert['metric']}\n"
                message += f"  Value: {alert['value']:.4f}\n"
                message += f"  Condition: {alert['condition']}\n"
                message += f"  Severity: {alert['severity']}\n"
                if alert['description']:
                    message += f"  {alert['description']}\n"
                message += f"  Time: {alert['timestamp']}\n\n"

        return message

    def _get_severity_color(self, severity: str) -> str:
        """
        Get color for severity level.

        Args:
            severity (str): Severity level.

        Returns:
            str: Color code.
        """
        colors = {
            'info': 'good',
            'warning': 'warning',
            'error': 'danger',
            'critical': '#FF0000'
        }
        return colors.get(severity, 'warning')

    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent alert history.

        Args:
            hours (int): Hours of history to retrieve.

        Returns:
            List[Dict[str, Any]]: Recent alerts.
        """
        if not self.alert_history:
            return []

        # Filter by time
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]

        return recent_alerts

    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get alert summary statistics.

        Returns:
            Dict[str, Any]: Alert summary.
        """
        if not self.alert_history:
            return {'total_alerts': 0, 'alerts_by_severity': {}, 'recent_alerts': 0}

        # Count by severity
        severity_counts = {}
        for alert in self.alert_history:
            severity = alert['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Recent alerts (last 24 hours)
        recent_alerts = len(self.get_alert_history(24))

        return {
            'total_alerts': len(self.alert_history),
            'alerts_by_severity': severity_counts,
            'recent_alerts': recent_alerts
        }


def create_default_energy_alert_rules() -> List[AlertRule]:
    """
    Create default alert rules for energy demand forecasting.

    Returns:
        List[AlertRule]: Default alert rules.
    """
    rules = [
        AlertRule(
            name="High Peak Demand Error",
            metric="peak_demand.peak_mape",
            condition=">",
            threshold=15.0,
            severity="error",
            description="Peak demand forecast error exceeds 15%"
        ),
        AlertRule(
            name="Low Model Accuracy",
            metric="mae",
            condition=">",
            threshold=300.0,
            severity="warning",
            description="Model MAE exceeds acceptable threshold"
        ),
        AlertRule(
            name="Data Drift Detected",
            metric="data_drift.drift_score",
            condition=">",
            threshold=0.05,
            severity="warning",
            description="Significant data drift detected"
        ),
        AlertRule(
            name="Low Reliability Score",
            metric="forecast_reliability.reliability_score",
            condition="<",
            threshold=70.0,
            severity="warning",
            description="Forecast reliability score below 70%"
        ),
        AlertRule(
            name="High Business Impact",
            metric="business_impact.business_impact_score",
            condition="<",
            threshold=80.0,
            severity="error",
            description="Business impact score below 80%"
        ),
        AlertRule(
            name="Low Renewable Penetration",
            metric="renewable_contribution.renewable_penetration",
            condition="<",
            threshold=20.0,
            severity="info",
            description="Renewable energy penetration below 20%"
        )
    ]

    return rules


if __name__ == "__main__":
    # Example usage
    alert_manager = AlertManager()

    # Add default rules
    default_rules = create_default_energy_alert_rules()
    for rule in default_rules:
        alert_manager.add_rule(rule)

    # Sample metrics
    sample_metrics = {
        'mae': 250.0,
        'peak_demand': {'peak_mape': 12.0},
        'data_drift': {'drift_score': 0.03},
        'forecast_reliability': {'reliability_score': 85.0},
        'business_impact': {'business_impact_score': 75.0},
        'renewable_contribution': {'renewable_penetration': 25.0}
    }

    # Evaluate rules
    alerts = alert_manager.evaluate_all_rules(sample_metrics)

    print(f"Triggered {len(alerts)} alerts:")
    for alert in alerts:
        print(f"- {alert['rule_name']}: {alert['description']}")
