import argparse
import yaml
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output", type=str, default=None)
    parser.add_argument("--reference_output", type=str, default=None)
    parser.add_argument("--model_generate", action="store_true", default=False)
    parser.add_argument("--ref_generate", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    #从config文件读取 judge_method
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    judge_method = config["judge_method"]
    ranking_prompt_path = config["prompts"][judge_method]
    ranking_prompt = open(ranking_prompt_path, "r").read()

    if args.model_generate:
        assert args.model_output is not None
        # generate and save to args.model_output
        pass
    if args.ref_generate:
        assert args.reference_output is not None
        # generate and save to args.reference_output
        pass
    if args.eval:
        if judge_method == "ranking":
            assert args.model_output is not None
            assert args.reference_output is not None
            ranking_prompt = config["prompt"]["ranking"]
        # load model_output and reference_output, do evaluation
        pass
