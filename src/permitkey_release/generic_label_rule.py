"""Withhold solar PV and EV charger labels on generic administrative text (release rule)."""
import json
from pathlib import Path

RULE = json.loads(Path(__file__).with_name('generic_label_rule.json').read_text(encoding='utf-8'))
GENERIC = set(RULE['generic_labels'])
CATEGORIES = RULE['categories']
FIELDS = ('description', 'native_type', 'native_subtype')


def generic_only(record):
    """True when every available text field is blank or a listed generic label."""
    return all(str(record.get(k) or '').strip().lower() in GENERIC for k in FIELDS)


def apply(record, predictions):
    """Return predictions with solar PV / EV charger set to 0 for generic-only records."""
    if generic_only(record):
        predictions = dict(predictions)
        for k in CATEGORIES:
            if k in predictions:
                predictions[k] = 0
    return predictions
