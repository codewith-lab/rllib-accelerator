import argparse
import itertools
import os
import sys

import ray

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from teacher_student.config import DEFAULT_TEACHER_STUDENT_PARAMS
from teacher_student.trainer import TeacherStudentTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multiple teacher-student experiments varying student size."
    )
    parser.add_argument("--env-id", type=str, default=DEFAULT_TEACHER_STUDENT_PARAMS["env_id"])
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_TEACHER_STUDENT_PARAMS["num_epochs"])
    parser.add_argument(
        "--train-batch-size", type=int, default=DEFAULT_TEACHER_STUDENT_PARAMS["train_batch_size"]
    )
    parser.add_argument(
        "--rollout-fragment-length",
        type=int,
        default=DEFAULT_TEACHER_STUDENT_PARAMS["rollout_fragment_length"],
    )
    parser.add_argument(
        "--num-rollout-workers",
        type=int,
        default=DEFAULT_TEACHER_STUDENT_PARAMS["num_rollout_workers"],
    )
    parser.add_argument("--student-lr", type=float, default=DEFAULT_TEACHER_STUDENT_PARAMS["student_lr"])
    parser.add_argument(
        "--student-hidden-dims",
        type=int,
        nargs="+",
        default=[DEFAULT_TEACHER_STUDENT_PARAMS["student_hidden_dim"]],
        help="List of student hidden dimensions to sweep.",
    )
    parser.add_argument(
        "--student-hidden-depths",
        type=int,
        nargs="+",
        default=[DEFAULT_TEACHER_STUDENT_PARAMS["student_hidden_depth"]],
        help="List of student hidden depths to sweep.",
    )
    parser.add_argument("--teacher-lr", type=float, default=DEFAULT_TEACHER_STUDENT_PARAMS["teacher_lr"])
    parser.add_argument(
        "--teacher-hidden-dim",
        type=int,
        default=DEFAULT_TEACHER_STUDENT_PARAMS["teacher_hidden_dim"],
    )
    parser.add_argument(
        "--teacher-hidden-depth",
        type=int,
        default=DEFAULT_TEACHER_STUDENT_PARAMS["teacher_hidden_depth"],
    )
    parser.add_argument("--clip-param", type=float, default=DEFAULT_TEACHER_STUDENT_PARAMS["clip_param"])
    parser.add_argument(
        "--value-loss-coeff",
        type=float,
        default=DEFAULT_TEACHER_STUDENT_PARAMS["value_loss_coeff"],
    )
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=DEFAULT_TEACHER_STUDENT_PARAMS["entropy_coeff"],
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=DEFAULT_TEACHER_STUDENT_PARAMS["max_grad_norm"],
    )
    parser.add_argument("--device", type=str, default=DEFAULT_TEACHER_STUDENT_PARAMS["device"])
    parser.add_argument(
        "--log-dir",
        type=str,
        default=DEFAULT_TEACHER_STUDENT_PARAMS["log_dir"],
        help="Directory to store JSONL logs for every run.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="teacher_student",
        help="Prefix for naming each run in logs.",
    )
    parser.add_argument(
        "--teacher-eval-episodes",
        type=int,
        default=DEFAULT_TEACHER_STUDENT_PARAMS["teacher_eval_episodes"],
        help="Number of greedy eval episodes for the teacher after each epoch (0 disables).",
    )
    return parser.parse_args()


def build_base_params(args):
    return {
        "env_id": args.env_id,
        "num_epochs": args.num_epochs,
        "train_batch_size": args.train_batch_size,
        "rollout_fragment_length": args.rollout_fragment_length,
        "num_rollout_workers": args.num_rollout_workers,
        "student_lr": args.student_lr,
        "teacher_lr": args.teacher_lr,
        "teacher_hidden_dim": args.teacher_hidden_dim,
        "teacher_hidden_depth": args.teacher_hidden_depth,
        "clip_param": args.clip_param,
        "value_loss_coeff": args.value_loss_coeff,
        "entropy_coeff": args.entropy_coeff,
        "max_grad_norm": args.max_grad_norm,
        "device": args.device,
        "log_dir": args.log_dir,
        "student_use_residual": DEFAULT_TEACHER_STUDENT_PARAMS["student_use_residual"],
        "teacher_use_residual": DEFAULT_TEACHER_STUDENT_PARAMS["teacher_use_residual"],
        "teacher_eval_episodes": args.teacher_eval_episodes,
    }


def main():
    args = parse_args()
    combos = list(itertools.product(args.student_hidden_dims, args.student_hidden_depths))
    if not combos:
        raise ValueError("Provide at least one student hidden dim/depth combination.")

    ray.init(include_dashboard=False)
    results = []
    try:
        for idx, (dim, depth) in enumerate(combos, start=1):
            hparams = build_base_params(args)
            hparams["student_hidden_dim"] = dim
            hparams["student_hidden_depth"] = depth
            run_name = f"{args.run_prefix}_dim{dim}_depth{depth}"
            hparams["run_name"] = run_name
            print(f"\n[TeacherStudentGrid] === Run {idx}/{len(combos)}: {run_name} ===")
            trainer = TeacherStudentTrainer(hparams)
            try:
                trainer.run(hparams["num_epochs"])
                trainer.summary()
                results.append({"run_name": run_name, "log_path": trainer.log_path})
            finally:
                trainer.shutdown()
    finally:
        ray.shutdown()

    print("\n[TeacherStudentGrid] All runs completed. Logs:")
    for rec in results:
        print(f"- {rec['run_name']}: {rec['log_path']}")


if __name__ == "__main__":
    main()
