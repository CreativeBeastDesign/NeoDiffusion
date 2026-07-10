# Deciding on toy config

## GQA-ratio
1. Find num_attention_heads & num_key_value_heads in config.json 
2. Calculate GQA ratio: num_attention_heads / num_key_value_heads
3. Apply a lower number with same ratio
