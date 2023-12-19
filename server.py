from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi

import numpy as np

kvCache = None

def write_kv_to_file(idx, content):
    overhead = 416
    embd = 1024
    layer = 32
    ctx = 512
    wtype = 2
    tokens = 4
    k_sz = wtype * embd * tokens
    v_sz_per_embd = tokens * wtype
    if len(content) != k_sz * 2:
        return
    assert len(content) == k_sz * 2
    content = bytearray(content)
    kt = np.array(content[:k_sz], dtype=np.int8)
    vt = np.array(content[k_sz:], dtype=np.int8)
    kt = kt.reshape((tokens, wtype * embd))
    vt = vt.reshape((embd, wtype * tokens))
    print(kt.sum(axis=1), vt.sum(axis=0), flush=True)

    unit = embd * ctx * wtype + overhead

    global kvCache

    if idx == 0:
        sz = unit * 2 * layer
        print("alloc size:", sz, flush=True)
        kvCache = bytearray([0] * sz)
    # k
    k = unit * 2 * idx + overhead
    kvCache[k: k + k_sz] = content[:k_sz]
    # v
    v = k + unit
    x = wtype * embd
    for x in range(embd):
        u = v + x * ctx * wtype
        kvCache[u: u + v_sz_per_embd] = content[k_sz + x * v_sz_per_embd: k_sz + (x + 1) * v_sz_per_embd]

    if idx == layer - 1:
        with open('kv.bin', 'wb') as f:
            print(len(kvCache), flush=True)
            f.write(kvCache)
        


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        # Parse the form data posted
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself
        
        # Print the POST request's body to the console/screen
        print(f"Received: {len(post_data)} bytes, content_length = {content_length}, kv_cache = {self.headers.get('kvCache', 'N/A')}")
        # write_kv_to_file(int(self.headers['kvCache'][2:]), post_data)

        # Send response status code
        self.send_response(200)

        # Send headers
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        # Send response content
        response_content = b"POST request received"
        self.wfile.write(response_content)


def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting HTTP server on port {port}...")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
