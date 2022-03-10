import json

data = {
    'users' : [
        {
            'name' : 'unknown',
            'fld_name' : 'unknown'
        }
    ]
}

with open(r'storage\something\users.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
