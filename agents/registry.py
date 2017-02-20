from adv_ac import get_adv_ac_agent_builder

def get_agent_builder(agent_id):
    parts = agent_id.split('.')
    rest = '.'.join(parts[1:])
    if parts[0] == 'adv_ac':
        return get_adv_ac_agent_builder(rest)
