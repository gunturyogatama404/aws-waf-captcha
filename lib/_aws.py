import base64, random, ast, time
from ._ai import get_solutions
from ._models import TokenTask, Proxy
from curl_cffi.requests import Session
from curl_cffi.requests.errors import RequestsError

class InvalidCaptchaError(Exception): pass
class InvalidProxyError(Exception): pass

class AwsTokenProcessor:
    def __init__(self, task: TokenTask):
        proxy, proxy_auth = None, None

        if isinstance(task.proxy, Proxy):
            proxy = f"http://{task.proxy.host}:{task.proxy.port}"
            if task.proxy.username and task.proxy.password:
                proxy_auth = (task.proxy.username, task.proxy.password)

        elif isinstance(task.proxy, str):
            parts = task.proxy.split(':')
            if len(parts) not in (2, 4):
                raise InvalidProxyError("Invalid proxy format")
            proxy = f"http://{parts[0]}:{parts[1]}"
            if len(parts) == 4:
                proxy_auth = (parts[2], parts[3])

        self.session = Session(proxy=proxy, proxy_auth=proxy_auth, verify=False)
        self.token_url = f"{task.token_url}/voucher"
        self.captcha_url = f"{task.captcha_url}/verify"
        self.problem_url = f"{task.captcha_url}/problem?kind=visual&domain={task.site_host}&locale=en-gb"

        self.goku_props = ast.literal_eval(base64.b64decode(task.goku_props).decode())
        self._get_challenge()

    def _get_challenge(self):
        try:
            challenge = self.session.get(self.problem_url, impersonate='chrome124', timeout=15).json()
        except RequestsError as e:
            raise InvalidProxyError(f"Network error: {e}")

        try:
            self.payload = challenge['state']['payload']
            self.iv = challenge['state']['iv']
            self.hmac_tag = challenge['hmac_tag']
            self.key = challenge['key']
            self.images = ast.literal_eval(challenge['assets']['images'])
            self.target = ast.literal_eval(challenge['assets']['target'])[0]
        except Exception:
            raise InvalidCaptchaError("Invalid CAPTCHA structure")

    def get_token(self):
        start = time.time()
        solutions = get_solutions(self.images, self.target)
        elapsed = int((time.time() - start) * 1000)

        verify_data = {
            "state": {"iv": self.iv, "payload": self.payload},
            "key": self.key,
            "hmac_tag": self.hmac_tag,
            "client_solution": solutions,
            "metrics": {"solve_time_millis": max(3000, min(9000, elapsed))},
            "goku_props": self.goku_props,
            "locale": "en-gb"
        }

        resp = self.session.post(self.captcha_url, json=verify_data, impersonate='chrome124').json()
        if not resp.get('success'):
            raise InvalidCaptchaError("CAPTCHA failed")

        token_req = {"captcha_voucher": resp['captcha_voucher'], "existing_token": None}
        final = self.session.post(self.token_url, json=token_req, impersonate='chrome124').json()
        return final.get('token')
