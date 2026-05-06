from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.payload_config import CONFIG


TEMPLATES = {
    "fixed": ROOT / "models" / "mobile_base_payload_fixed.xml",
    "free": ROOT / "models" / "mobile_base_payload_free.xml",
}

OUTPUTS = {
    "fixed": ROOT / "models" / "scene_fixed.xml",
    "free": ROOT / "models" / "scene.xml",
}


def generate_model(template_path, output_path, config):
    with open(template_path, "r") as f:
        template = f.read()

    filled_xml = template.format(**config)

    with open(output_path, "w") as f:
        f.write(filled_xml)

    print(f"Generated {output_path}")


def generate_all_models(config=None):
    if config is None:
        config = CONFIG

    for name in TEMPLATES:
        generate_model(
            template_path=TEMPLATES[name],
            output_path=OUTPUTS[name],
            config=config,
        )


def main():
    generate_all_models()


if __name__ == "__main__":
    main()