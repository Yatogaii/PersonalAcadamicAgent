from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape
import os

# Initialize Jinja2 environment
env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__)),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)

def apply_prompt_template(prompt_name: str, params: dict = {}) -> list:
    """
    Apply template variables to a prompt template and return formatted messages.

    Args:
        prompt_name: Name of the prompt template to use

    Returns:
        List of messages with the system prompt as the first message
    """
    try:
        template = env.get_template(f"{prompt_name}.md")
        system_prompt = template.render(**params)
        return [{"role": "system", "content": system_prompt}]
    except Exception as e:
        raise ValueError(f"Error applying template {prompt_name}: {e}")
