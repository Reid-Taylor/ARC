#!/usr/bin/env python3
"""
Script to register Flyte workflows and launch training jobs.
"""
import subprocess
import sys
import argparse
import os
from pathlib import Path


def run_command(command: str, cwd: str = None) -> tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    print(f"Running: {command}")
    result = subprocess.run(
        command, 
        shell=True, 
        capture_output=True, 
        text=True,
        cwd=cwd
    )
    return result.returncode, result.stdout, result.stderr


def build_and_push_docker_image(registry: str, tag: str = "latest"):
    """Build and push Docker image for Flyte workflows."""
    print("Building Docker image for Flyte workflows...")
    
    # Build image
    build_cmd = f"docker build -f Dockerfile.flyte -t {registry}/arc-encoder-training:{tag} ."
    exit_code, stdout, stderr = run_command(build_cmd)
    
    if exit_code != 0:
        print(f"Error building Docker image: {stderr}")
        return False
    
    print("Docker image built successfully!")
    
    # Push image
    print("Pushing Docker image to registry...")
    push_cmd = f"docker push {registry}/arc-encoder-training:{tag}"
    exit_code, stdout, stderr = run_command(push_cmd)
    
    if exit_code != 0:
        print(f"Error pushing Docker image: {stderr}")
        return False
    
    print("Docker image pushed successfully!")
    return True


def register_workflow(
    project: str = "arc-training",
    domain: str = "development", 
    config_file: str = "flyte_workflows/config.yaml"
):
    """Register Flyte workflow with the cluster."""
    print("Registering Flyte workflow...")
    
    workflow_path = "flyte_workflows/arc_encoder_training.py"
    
    register_cmd = (
        f"pyflyte register "
        f"--project {project} "
        f"--domain {domain} "
        f"--config {config_file} "
        f"{workflow_path}"
    )
    
    exit_code, stdout, stderr = run_command(register_cmd)
    
    if exit_code != 0:
        print(f"Error registering workflow: {stderr}")
        return False
    
    print("Workflow registered successfully!")
    print(stdout)
    return True


def launch_workflow(
    project: str = "arc-training",
    domain: str = "development",
    config_file: str = "flyte_workflows/config.yaml",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    alpha: float = 0.85,
    dataset_path: str = "training"
):
    """Launch training workflow on Flyte cluster."""
    print("Launching training workflow...")
    
    launch_cmd = (
        f"pyflyte run "
        f"--project {project} "
        f"--domain {domain} "
        f"--config {config_file} "
        f"flyte_workflows/arc_encoder_training.py "
        f"arc_encoder_training_workflow "
        f"--dataset_path {dataset_path} "
        f"--epochs {epochs} "
        f"--batch_size {batch_size} "
        f"--learning_rate {learning_rate} "
        f"--alpha {alpha}"
    )
    
    exit_code, stdout, stderr = run_command(launch_cmd)
    
    if exit_code != 0:
        print(f"Error launching workflow: {stderr}")
        return False
    
    print("Workflow launched successfully!")
    print(stdout)
    return True


def local_test():
    """Test workflow locally using pyflyte run."""
    print("Testing workflow locally...")
    
    test_cmd = (
        "pyflyte run "
        "--config flyte_workflows/config-local.yaml "
        "flyte_workflows/arc_encoder_training.py "
        "arc_encoder_training_workflow "
        "--dataset_path training "
        "--epochs 2 "
        "--batch_size 4 "
        "--learning_rate 1e-3 "
        "--alpha 0.85"
    )
    
    exit_code, stdout, stderr = run_command(test_cmd)
    
    if exit_code != 0:
        print(f"Error running local test: {stderr}")
        return False
    
    print("Local test completed successfully!")
    print(stdout)
    return True


def main():
    """Main function to handle command line arguments and execute actions."""
    parser = argparse.ArgumentParser(description="Flyte workflow management for ARC Encoder")
    
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # Build and push image
    build_parser = subparsers.add_parser("build", help="Build and push Docker image")
    build_parser.add_argument("--registry", required=True, help="Docker registry URL")
    build_parser.add_argument("--tag", default="latest", help="Docker image tag")
    
    # Register workflow
    register_parser = subparsers.add_parser("register", help="Register workflow with Flyte")
    register_parser.add_argument("--project", default="arc-training", help="Flyte project")
    register_parser.add_argument("--domain", default="development", help="Flyte domain")
    register_parser.add_argument("--config", default="flyte_workflows/config.yaml", help="Flyte config file")
    
    # Launch workflow
    launch_parser = subparsers.add_parser("launch", help="Launch training workflow")
    launch_parser.add_argument("--project", default="arc-training", help="Flyte project")
    launch_parser.add_argument("--domain", default="development", help="Flyte domain")
    launch_parser.add_argument("--config", default="flyte_workflows/config.yaml", help="Flyte config file")
    launch_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    launch_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    launch_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    launch_parser.add_argument("--alpha", type=float, default=0.85, help="Alpha parameter")
    launch_parser.add_argument("--dataset-path", default="training", help="Dataset path")
    
    # Local test
    test_parser = subparsers.add_parser("test", help="Test workflow locally")
    
    # Deploy (build + register + launch)
    deploy_parser = subparsers.add_parser("deploy", help="Full deployment (build + register + launch)")
    deploy_parser.add_argument("--registry", required=True, help="Docker registry URL")
    deploy_parser.add_argument("--tag", default="latest", help="Docker image tag")
    deploy_parser.add_argument("--project", default="arc-training", help="Flyte project")
    deploy_parser.add_argument("--domain", default="development", help="Flyte domain")
    deploy_parser.add_argument("--config", default="flyte_workflows/config.yaml", help="Flyte config file")
    deploy_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    deploy_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    deploy_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    deploy_parser.add_argument("--alpha", type=float, default=0.85, help="Alpha parameter")
    deploy_parser.add_argument("--dataset-path", default="training", help="Dataset path")
    
    args = parser.parse_args()
    
    if args.action == "build":
        success = build_and_push_docker_image(args.registry, args.tag)
        sys.exit(0 if success else 1)
    
    elif args.action == "register":
        success = register_workflow(args.project, args.domain, args.config)
        sys.exit(0 if success else 1)
    
    elif args.action == "launch":
        success = launch_workflow(
            args.project, args.domain, args.config,
            args.epochs, args.batch_size, args.learning_rate, args.alpha, args.dataset_path
        )
        sys.exit(0 if success else 1)
    
    elif args.action == "test":
        success = local_test()
        sys.exit(0 if success else 1)
    
    elif args.action == "deploy":
        # Full deployment pipeline
        print("Starting full deployment pipeline...")
        
        # Step 1: Build and push image
        if not build_and_push_docker_image(args.registry, args.tag):
            sys.exit(1)
        
        # Step 2: Register workflow
        if not register_workflow(args.project, args.domain, args.config):
            sys.exit(1)
        
        # Step 3: Launch workflow
        if not launch_workflow(
            args.project, args.domain, args.config,
            args.epochs, args.batch_size, args.learning_rate, args.alpha, args.dataset_path
        ):
            sys.exit(1)
        
        print("Full deployment completed successfully!")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()