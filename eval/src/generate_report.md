<!-- source: generate_report.ipynb -->

```python
import sys
import json
import pandas as pd
from dotenv import load_dotenv

from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential


load_dotenv()
%load_ext autoreload
%autoreload 2
sys.path.append("../../../")
```

```python
ACCOUNT_NAME = 'your_cosmos_account'
DATABASE_NAME = 'state'
CONTAINER_NAME = 'issues'



def get_issues_df_from_db(account_name, db_name, container_name):
    # Build the Cosmos DB account URL
    account_url = f"https://{account_name}.documents.azure.com:443/"

    # Use DefaultAzureCredential for authentication
    credential = DefaultAzureCredential()
    client = CosmosClient(account_url, credential=credential)

    # Access database and container
    database = client.get_database_client(db_name)
    container = database.get_container_client(container_name)

    # Query and process data
    data = list()
    items = container.query_items(
            query='SELECT * FROM c',
            enable_cross_partition_query=True)
    for item in items:
        data.append(json.dumps(item, indent=True))

    # Convert to DataFrame
    df = pd.DataFrame.from_dict([json.loads(i) for i in data])

    return df
```

```python
data = get_issues_df_from_db(ACCOUNT_NAME, DATABASE_NAME, CONTAINER_NAME)
```

```python
data.head(10)
```

```python
from system_monitor import SystemMonitor
calculator = SystemMonitor("../config.json")
```

```python
data = get_issues_df_from_db(ACCOUNT_NAME, DATABASE_NAME, CONTAINER_NAME)
calculator.calculate_metrics(data)
print(calculator.get_acceptance_rate())
print(calculator.get_amount_of_reviewed_documents())
print(calculator.get_issue_type_distribution())
```
