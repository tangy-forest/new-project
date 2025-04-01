from http.server import HTTPServer, BaseHTTPRequestHandler
 import subprocess
 import os
 
 class StreamlitHandler(BaseHTTPRequestHandler):
     def do_GET(self):
         # Set PYTHONPATH to include the parent directory
         env = os.environ.copy()
         env['PYTHONPATH'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
         
         # Start Streamlit process
         process = subprocess.Popen(
             ['streamlit', 'run', '../src/app.py'],
             env=env,
             stdout=subprocess.PIPE,
             stderr=subprocess.PIPE
         )
         
         # Get the output
         stdout, stderr = process.communicate()
         
         # Send response
         self.send_response(200)
         self.send_header('Content-type', 'text/html')
         self.end_headers()
         
         # Write the Streamlit output
         self.wfile.write(stdout)
 
 def handler(event, context):
     server = HTTPServer(('', 8000), StreamlitHandler)
     return server.serve_forever() 