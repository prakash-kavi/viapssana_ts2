from config.meditation_config import load_actinf_params_from_json

print('novice ->', type(load_actinf_params_from_json(None, 'novice')))
print('expert ->', type(load_actinf_params_from_json(None, 'expert')))
