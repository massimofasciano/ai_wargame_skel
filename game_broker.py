from http.server import HTTPServer, BaseHTTPRequestHandler
from http import HTTPStatus
import json
import random
import string
import socket   
import sys

def get_random_string(length):
    letters = string.ascii_lowercase + string.ascii_uppercase + string.digits
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

stored_information = None

server_id = get_random_string(8)
if len(sys.argv) > 1:
    server_id = sys.argv[1]

def remove_prefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text

invalid_id = {'success': False, 'error':'invalid server id'}
return_information = {'success': True, 'data':stored_information}

class _RequestHandler(BaseHTTPRequestHandler):

    def _set_ok(self):
        self.send_response(HTTPStatus.OK.value)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def _set_error(self):
        self.send_response(HTTPStatus.NOT_FOUND.value)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_GET(self):
        global server_id
        request_id = remove_prefix(self.path,'/')
        if request_id == server_id:
            global stored_information, return_information
            self._set_ok()
            return_information['data']=stored_information
            self.wfile.write(json.dumps(return_information).encode('utf-8'))
        else:
            global invalid_id
            self._set_error()
            self.wfile.write(json.dumps(invalid_id).encode('utf8'))

    def do_POST(self):
        global server_id
        request_id = remove_prefix(self.path,'/')
        if request_id == server_id:
            global stored_information, return_information
            content_length = self.headers.get('content-length')
            if content_length != None:
                length = int(content_length)
                stored_information = json.loads(self.rfile.read(length))
                self._set_ok()
                return_information['data']=stored_information
                self.wfile.write(json.dumps(return_information).encode('utf-8'))
                print(stored_information)
        else:
            global invalid_id
            self._set_error()
            self.wfile.write(json.dumps(invalid_id).encode('utf8'))

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT.value)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST')
        self.send_header('Access-Control-Allow-Headers', 'content-type')
        self.end_headers()

def run_server():
    hostname=socket.gethostname()   
    IPAddr=socket.gethostbyname(hostname)   
    server_address = ('', 8001)
    httpd = HTTPServer(server_address, _RequestHandler)
    print(f'serving at http://{IPAddr}:{server_address[1]}/{server_id}')
    httpd.serve_forever()


if __name__ == '__main__':
    run_server()
