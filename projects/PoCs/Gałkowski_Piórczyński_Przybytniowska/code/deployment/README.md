## Running the Embedder and LLM on a Cluster

Both the `embedder` and `llm` folders should be run on a cluster. Please update the `.sh` scripts with your information before running (replace **<FILL_THIS>** and update the conda path). After running on SLURM, ports need to be forwarded to call APIs from your local computer.

### Port Forwarding

#### Embedder
Check which DGX your job is running on and replace the value below:
```bash
ssh -L 8080:dgx-4:8080 -J <mini_server_login>@ssh.mini.pw.edu.pl <eden_server_login>@eden.mini.pw.edu.pl
```

#### LLM
Check which DGX your job is running on and replace the value below:
```bash
ssh -L 8000:dgx-4:8000 -J <mini_server_login>@ssh.mini.pw.edu.pl <eden_server_login>@eden.mini.pw.edu.pl
```

### API Examples

#### Calling Embedder
```bash
curl -X POST "http://0.0.0.0:8080/embeddings" -H "Content-Type: application/json" -d '{"messages": "Kiedy zostanie opublikowany plan zajec na semestr letni?"}'
```

#### Calling LLM
The `vllm serve` command creates a REST API in the same format as the OpenAI API ([docs](https://platform.openai.com/docs/api-reference)):

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [
    {"role": "user", "content": "Kiedy zostanie opublikowany plan zajec na semestr letni?"}
  ]
}'
```
