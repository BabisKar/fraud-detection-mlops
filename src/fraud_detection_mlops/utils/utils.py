import yaml
import logging
import fraud_detection_mlops
import importlib_resources

def load_config(logger: logging.Logger, yaml_file_name:str) -> dict:
    """
    Loads a YAML configuration file and returns the configuration for the specified environment.
    Args:
        logger: Logger object for logging messages.
        yaml_file_name (str): Name of the YAML file to load.
        env (str): The environment key to extract the configuration for.

    Returns:
        dict: Configuration for the specified script.
    """
    # Construct the reference to the YAML file
    ref = importlib_resources.files(fraud_detection_mlops) / f"configs/{yaml_file_name}"

    # Use the file reference to access the file
    with importlib_resources.as_file(ref) as config_path:
        logger.info(f"Loading config from {config_path}")

        # Open and read the YAML file
        with open(config_path, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file)

    # Return the configuration for the specified environment
    return config_data