import re
import ast

def parse_logs():
    log_file = "logs/search.log"

    # Updated regex to match the new log format
    log_pattern = re.compile(
        r"(?P<time>[^\s]+) \| (?P<level>[A-Z]+) \| (?P<message>.+?) \| (?P<extra>\{.*\})"
    )

    parsed_logs = []

    with open(log_file, "r") as f:
        for line in f:
            match = log_pattern.match(line)
            if match:
                log_data = match.groupdict()
                extra_str = log_data["extra"]

                try:
                    log_data["extra"] = ast.literal_eval(extra_str)  # Safely parse dictionary
                    parsed_logs.append(log_data)
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing log entry: {extra_str}")
                    print(f"Parsing Error: {e}")

    # Example: Print parsed logs
    for log in parsed_logs:
        print(log)

if __name__ == "__main__":
    parse_logs()
