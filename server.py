from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        # Parse the form data posted
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself
        
        # Print the POST request's body to the console/screen
        print(f"Received: {len(post_data)} bytes, content_length = {content_length}, kv_cache = {self.headers.get('kvCache', 'N/A')}")

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
