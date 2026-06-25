import argparse
from model_loader import LiverSegmentator


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)

    args = parser.parse_args()

    model = LiverSegmentator(args.model)

    model.predict(
        input_folder=args.input,
        output_folder=args.output
    )


if __name__ == "__main__":
    main()