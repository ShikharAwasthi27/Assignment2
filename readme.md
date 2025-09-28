# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (first time)
python cli.py --train --data assignment2data.csv

# 3. Get recommendations
python cli.py --profile "I've completed Python and want data visualization" --completed 101

# 4. Run full evaluation for your report
python test_profiles.py
