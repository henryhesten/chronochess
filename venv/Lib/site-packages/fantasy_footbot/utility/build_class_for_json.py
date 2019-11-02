import os
import re


def build_class_for_json(json_data, class_name, init_line_break='\n        '):
    class_template_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "auto_json_class_template.txt")
    with open(class_template_path, 'r') as class_template_file:
        class_template = class_template_file.read()
    top_level_attributes = list(json_data.keys())
    top_level_attributes.sort()
    init_assignments = ["self.{a} = json_data['{a}']".format(a=attribute) for attribute in top_level_attributes]
    json_data_class = class_template.format(class_name=class_name, init_body=init_line_break.join(init_assignments))
    with open('{}.py'.format(to_snake_case(class_name)), 'w') as new_class_file:
        new_class_file.write(json_data_class)


def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
