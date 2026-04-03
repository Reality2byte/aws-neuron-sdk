#!/bin/bash
# Setup script to fetch Jira API token from AWS Secrets Manager
# and configure it for the Atlassian MCP server

set -e

echo "Setting up Jira API token..."

# Check if AWS CLI is available
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed"
    echo "Install with: brew install awscli"
    exit 1
fi

# Check if ada is available
if ! command -v ada &> /dev/null; then
    echo "Error: ada credentials tool is not installed"
    echo "Install with: toolbox install ada"
    exit 1
fi

# Set AWS profile to kaena
export AWS_PROFILE=kaena

echo "Fetching Jira API token from AWS Secrets Manager..."
JIRA_TOKEN=$(aws secretsmanager get-secret-value \
    --secret-id NKI_JIRA_API_TOKEN \
    --region us-west-2 \
    --query SecretString \
    --output text 2>&1)

if [ $? -ne 0 ]; then
    echo "Error: Failed to fetch Jira API token"
    echo "Make sure you have:"
    echo "  1. Run 'ada credentials setup' with account 621547421844, role Admin, profile kaena"
    echo "  2. Added kaena profile to ~/.aws/config with ada credential_process"
    echo "  3. Have IAM permissions to access the secret"
    echo ""
    echo "Error details:"
    echo "$JIRA_TOKEN"
    exit 1
fi

echo "✓ Successfully fetched Jira API token"

# Update the MCP config with the actual token
MCP_CONFIG="$HOME/.kiro/settings/mcp.json"

if [ ! -f "$MCP_CONFIG" ]; then
    echo "Error: MCP config not found at $MCP_CONFIG"
    exit 1
fi

# Create a temporary file with the token substituted
python3 << EOF
import json
import os

config_path = os.path.expanduser('$MCP_CONFIG')
with open(config_path, 'r') as f:
    config = json.load(f)

# Update the Jira API token
if 'atlassian-jira' in config['mcpServers']:
    config['mcpServers']['atlassian-jira']['env']['JIRA_API_TOKEN'] = '''$JIRA_TOKEN'''
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Updated MCP configuration with Jira API token")
else:
    print("Error: atlassian-jira server not found in MCP config")
    exit(1)
EOF

echo ""
echo "Setup complete! You can now use Jira tools in Kiro."
echo ""
echo "To use Jira MCP tools:"
echo "  1. Restart Kiro CLI"
echo "  2. Use Jira tools through the MCP server"
echo ""
echo "Example queries:"
echo "  - Search for NKI tickets"
echo "  - Get ticket details"
echo "  - Create new tickets"
