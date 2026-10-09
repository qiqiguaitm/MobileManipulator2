#!/usr/bin/env python3
"""Interactive secret entry on robot02; no secret in argv, logs or the repository."""
import getpass
import json
import os
from pathlib import Path
import tempfile


def main():
    path = Path.home()/'.config/robot-object-approach/openai.json'
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    key = getpass.getpass('OpenAI API key（隐藏输入，不回显）: ').strip()
    if not key or any(c.isspace() for c in key):
        raise SystemExit('密钥为空或包含空白，未保存。')
    model = input('模型名称 [gpt-4.1]: ').strip() or 'gpt-4.1'
    fd, name = tempfile.mkstemp(prefix='.openai-', dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(dict(api_key=key, model=model), f)
        os.replace(name,path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    print('配置已保存（权限 600）。下一步运行 go_to_object.sh --check 做静态测试。')


if __name__ == '__main__':
    main()
