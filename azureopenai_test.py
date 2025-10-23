import asyncio
from openai import AsyncAzureOpenAI

# --- Replace these with your actual details ---
AZURE_OPENAI_ENDPOINT = "https://YOUR-RESOURCE-NAME.openai.azure.com/"
AZURE_OPENAI_KEY = "YOUR-API-KEY"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"  # Replace with your deployed model name
API_VERSION = "2024-05-01-preview"

# --- Async test function ---
async def test_azure_openai():
    client = AsyncAzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=API_VERSION,
    )

    # Send a simple prompt
    response = await client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello from Azure OpenAI!"}
        ],
    )

    print(" Connected successfully!")
    print("Response:", response.choices[0].message.content)

# --- Run the async test ---
if __name__ == "__main__":
    asyncio.run(test_azure_openai())
