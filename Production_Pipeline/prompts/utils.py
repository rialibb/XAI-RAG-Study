import json


def JSON_metric_parser(raw_json, metric_names):
    """
    Extracts the faithfulness values from the raw JSON response.
    """
    raw_json = raw_json[raw_json.find("{") : raw_json.rfind("}") + 1]
    try:
        parsed_json = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    else:
        return {
            f"answer_{j}": {
                metric_name: parsed_json[f"answer_{j}"][metric_name]
                for metric_name in metric_names
            }
            for j in range(1, len(parsed_json) + 1)
        }
