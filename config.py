from yaml import load, Loader,safe_load
import os
def deep_update(mapping, *updating_mappings):
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


class Config:
    def __init__(self, config_path):
        default_config = safe_load(open("config.default.yaml"))
        self.config = {}
        self.config.update(default_config)
        with open(config_path, 'r') as f:
            config = safe_load(f)
            self.config = deep_update(self.config, config)

    def get(self, key, default=None):
        # map section.key to [section][key]
        keys = key.split('.')
        val = self.config
        for k in keys:
            if k in val:
                try:
                    val = val[k]
                except:
                    val = default
                    print(f"Error: {key} is not a valid key, val = {val}, default = {default}")
                    break
            else:
                val = default
                break
        # check environment variables
        if val is None or val == "":
            val = default
        env_key = [f"phonelm-{key.upper().replace('.', '_')}", f"phonelm-{key.upper().split('.')[-1]}"]
        for k in env_key:
            if k in os.environ:
                val = os.environ[k]
                break
        return val


if __name__ == "__main__":
    config = Config("config.yaml")
    print(config.get("model"))
    print(config.get("model.name", "default"))
    print(config.get("datasets.path", False))
