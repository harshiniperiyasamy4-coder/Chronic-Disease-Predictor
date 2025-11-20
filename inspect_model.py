import joblib
from pprint import pprint

print("🔍 Loading hypertension model...")
try:
    # Load the saved model
    model_path = "models/hypertension_model.joblib"
    model = joblib.load(model_path)
    print(f"✅ Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# ------------------------------
# Inspect the pipeline
# ------------------------------
print("\n📦 Pipeline Steps:")
try:
    print(list(model.named_steps.keys()))
except Exception as e:
    print(f"Error accessing pipeline steps: {e}")

# ------------------------------
# Get the preprocessor
# ------------------------------
pre = None
try:
    pre = model.named_steps.get("preprocessor")
    print("\nPreprocessor type:", type(pre))
except Exception as e:
    print(f"Error retrieving preprocessor: {e}")

# ------------------------------
# Inspect transformers in the preprocessor
# ------------------------------
if pre is not None:
    if hasattr(pre, "transformers"):
        print("\n🧱 ColumnTransformer.transformers:")
        pprint(pre.transformers)

    if hasattr(pre, "named_transformers_"):
        print("\n🔤 ColumnTransformer.named_transformers_:")
        pprint(pre.named_transformers_.keys())

# ------------------------------
# Check feature names in the pipeline and preprocessor
# ------------------------------
print("\n📋 Checking feature names used in training:")
for obj_name, obj in [("pipeline", model), ("preprocessor", pre)]:
    try:
        if hasattr(obj, "feature_names_in_"):
            print(f"➡ {obj_name}.feature_names_in_:")
            pprint(getattr(obj, "feature_names_in_"))
    except Exception as e:
        print(f"⚠️ Error checking {obj_name} feature_names_in_: {e}")

# ------------------------------
# Try to call get_feature_names_out
# ------------------------------
try:
    if pre is not None and hasattr(pre, "get_feature_names_out"):
        names = pre.get_feature_names_out()
        print(f"\n🧩 pre.get_feature_names_out() -> length {len(names)}")
        pprint(names)
    else:
        print("\n⚠️ get_feature_names_out not available.")
except Exception as e:
    print(f"⚠️ Error calling get_feature_names_out: {e}")

# ------------------------------
# Final message
# ------------------------------
print("\n✅ Inspection complete. You can now compare these columns with your Flask input fields.")
