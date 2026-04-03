# Jira Integration Quick Start

## Prerequisites Check

Run these commands to verify you have everything installed:

```bash
# Check AWS CLI
aws --version

# Check ada credentials tool
ada --version

# Check Python 3
python3 --version

# Check if uvx is available (for MCP server)
uvx --version
```

If any are missing, install them:
```bash
# AWS CLI
brew install awscli

# ada credentials tool
toolbox install ada

# uv (includes uvx)
brew install uv
```

## One-Time Setup

### 1. Configure Ada Credentials

```bash
ada credentials setup
```

When prompted:
- **Account**: 621547421844
- **Role**: Admin
- **Profile name**: kaena

### 2. Add Kaena Profile to AWS Config

```bash
echo '[profile kaena]
credential_process='$HOME'/.toolbox/bin/ada credentials print --profile=kaena' >> ~/.aws/config
```

### 3. Run the Setup Script

```bash
cd /path/to/aws-neuron-sdk-staging
chmod +x _utilities/setup_jira_token.sh
./_utilities/setup_jira_token.sh
```

This script will:
- Fetch the Jira API token from AWS Secrets Manager
- Update your MCP configuration with the token
- Verify everything is set up correctly

### 4. Restart Kiro

After running the setup script, restart Kiro CLI to load the new MCP server.

## Using Jira in Kiro

Once set up, you can use Kiro Powers to interact with Jira:

```bash
# In Kiro CLI, check available powers
kiro powers list

# Look for Atlassian/Jira related tools
```

## Manual Verification

To manually verify the setup worked:

```bash
# Check MCP config has Jira server
cat ~/.kiro/settings/mcp.json | grep -A 10 atlassian-jira

# Test AWS Secrets Manager access
export AWS_PROFILE=kaena
aws secretsmanager get-secret-value \
    --secret-id NKI_JIRA_API_TOKEN \
    --region us-west-2 \
    --query SecretString \
    --output text
```

## Troubleshooting

### "Error: Failed to fetch Jira API token"

1. Verify ada credentials are set up:
   ```bash
   ada credentials list
   ```

2. Check AWS profile is configured:
   ```bash
   cat ~/.aws/config | grep -A 2 kaena
   ```

3. Test AWS access:
   ```bash
   export AWS_PROFILE=kaena
   aws sts get-caller-identity
   ```

### "MCP server not loading"

1. Check uvx is installed:
   ```bash
   uvx --version
   ```

2. Manually test the MCP server:
   ```bash
   uvx mcp-server-atlassian
   ```

3. Check Kiro MCP logs (location varies by installation)

## What's Next

After setup, you can:
- Query NKI Jira tickets
- Create new tickets
- Update ticket status
- Search and filter tickets
- Generate reports

See the full guide at `.kiro/steering/jira.md` for detailed usage examples.
