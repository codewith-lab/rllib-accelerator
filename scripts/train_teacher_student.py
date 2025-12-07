import argparse
import os
import sys

import ray

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from teacher_student.config import DEFAULT_TEACHER_STUDENT_PARAMS
from teacher_student.trainer import TeacherStudentTrainer


def build_hparams(cli_args):
    params = dict(DEFAULT_TEACHER_STUDENT_PARAMS)
    overrides = {
        "env_id": cli_args.env_id,
        "num_epochs": cli_args.num_epochs,
        "train_batch_size": cli_args.train_batch_size,
        "rollout_fragment_length": cli_args.rollout_fragment_length,
        "num_rollout_workers": cli_args.num_rollout_workers,
        "student_lr": cli_args.student_lr,
        "student_hidden_dim": cli_args.student_hidden_dim,
        "student_hidden_depth": cli_args.student_hidden_depth,
        "teacher_lr": cli_args.teacher_lr,
        "teacher_hidden_dim": cli_args.teacher_hidden_dim,
        "teacher_hidden_depth": cli_args.teacher_hidden_depth,
        "clip_param": cli_args.clip_param,
        "value_loss_coeff": cli_args.value_loss_coeff,
        "entropy_coeff": cli_args.entropy_coeff,
        "max_grad_norm": cli_args.max_grad_norm,
        "device": cli_args.device,
        "log_dir": cli_args.log_dir,
        "teacher_eval_episodes": cli_args.teacher_eval_episodes,
    }
    params.update(overrides)
    return params


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train student-teacher PPO where the student collects rollouts."
    )
    parser.add_argument("--env-id", type=str, default=DEFAULT_TEACHER_STUDENT_PARAMS["env_id"])
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_TEACHER_STUDENT_PARAMS["num_epochs"])
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=DEFAULT_TEACHER_STUDENT_PARAMS["train_batch_size"],
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
        "--student-hidden-dim",
        type=int,
        default=DEFAULT_TEACHER_STUDENT_PARAMS["student_hidden_dim"],
    )
    parser.add_argument(
        "--student-hidden-depth",
        type=int,
        default=DEFAULT_TEACHER_STUDENT_PARAMS["student_hidden_depth"],
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
        help="Directory to store teacher-student JSONL logs.",
    )
    parser.add_argument(
        "--teacher-eval-episodes",
        type=int,
        default=DEFAULT_TEACHER_STUDENT_PARAMS["teacher_eval_episodes"],
        help="Number of greedy episodes to evaluate the teacher per epoch (0 disables).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hparams = build_hparams(args)
    ray.init(include_dashboard=False)
    trainer = TeacherStudentTrainer(hparams)
    try:
        trainer.run(hparams["num_epochs"])
        trainer.summary()
    finally:
        trainer.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
