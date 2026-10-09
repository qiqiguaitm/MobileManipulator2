#!/usr/bin/env python3
"""Interactive secret entry on robot02; no secret in argv, logs or the repository."""
import getpass
import json
import os
from pathlib import Path
import tempfile


def main():
    path = Path.home()/'.config/robot-object-approach/qwen.json'
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    key = getpass.getpass('Qwen API key（隐藏输入，不回显）: ').strip()
    if not key or any(c.isspace() for c in key):
        raise SystemExit('密钥为空或包含空白，未保存。')
    model = input('模型名称 [qwen3-vl-plus]: ').strip() or 'qwen3-vl-plus'
    region = input('API Key地域 beijing/singapore [beijing]: ').strip() or 'beijing'
    if region not in ('beijing','singapore'):raise SystemExit('地域无效，未保存。')
    fd, name = tempfile.mkstemp(prefix='.qwen-', dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(dict(api_key=key, model=model, region=region), f)
        os.replace(name,path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    print('配置已保存（权限 600）。下一步运行 go_to_object.sh --check --detector qwen 做静态测试。')


if __name__ == '__main__':
    main()
