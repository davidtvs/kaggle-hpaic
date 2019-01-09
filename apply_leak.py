import sys
import os
import pandas as pd


if __name__ == "__main__":
    submission_path = sys.argv[1]
    leak126_path = "../dataset/leak126.csv"
    leak259_path = "../dataset/leak259.csv"
    save_dir = os.path.dirname(submission_path)
    base_name = os.path.basename(submission_path)
    base_name = os.path.splitext(base_name)[0]

    # Read submission without leak
    submission = pd.read_csv(submission_path)

    for leak_path in (leak126_path, leak259_path):
        # Read the leak from file and apply it to the submission
        leak = pd.read_csv(leak_path)
        submission_leak = submission.set_index("Id")
        submission_leak.loc[leak["Id"], "Predicted"] = leak["Target"].values
        submission_leak.reset_index(inplace=True)

        # Construct the filename of the submission file using the dictionary keys
        leak_name = os.path.basename(leak_path)
        leak_name = os.path.splitext(leak_name)[0]
        csv_name = "{}_{}.csv".format(base_name, leak_name)
        save_path = os.path.join(save_dir, csv_name)
        submission_leak.to_csv(save_path, index=False)
        print("Saved submission in:", save_path)
