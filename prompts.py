from pathlib import Path
from jinja2 import Environment, FileSystemLoader

_TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"

env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR))

PROMPT_TEMPLATES = {
    "simple_en":        "simple.jinja",
    "improved_en_v1":   "improved_en_v1.jinja",
}


def render_prompt(template_name: str, **kwargs) -> str:
    template_file = PROMPT_TEMPLATES[template_name]
    template = env.get_template(template_file)
    return template.render(**kwargs)
