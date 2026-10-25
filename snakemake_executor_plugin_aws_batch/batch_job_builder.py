import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from snakemake_interface_common.exceptions import WorkflowError
from snakemake_interface_executor_plugins.jobs import JobExecutorInterface
from snakemake_executor_plugin_aws_batch.batch_client import BatchClient
from snakemake_executor_plugin_aws_batch.constant import (
    VALID_RESOURCES_MAPPING,
    BATCH_JOB_DEFINITION_TYPE,
    BATCH_JOB_PLATFORM_CAPABILITIES,
    BATCH_JOB_RESOURCE_REQUIREMENT_TYPE,
)


class BatchJobBuilder:
    def __init__(
        self,
        logger,
        job: JobExecutorInterface,
        envvars: dict,
        container_image: str,
        settings,
        job_command: str,
        batch_client: BatchClient,
    ):
        self.logger = logger
        self.job = job
        self.envvars = envvars
        self.container_image = container_image
        self.settings = settings
        self.job_command = job_command
        self.batch_client = batch_client
        self.created_job_defs = []
        self.container_path = "/tmp/repo"
        if self.job.resources.get("aws_batch_unique_workdir", False):
            with TemporaryDirectory() as temp_dir_name:
                self.container_unique_workdir = str(Path(self.container_path) / Path(temp_dir_name).name)

    def _make_container_command(self, remote_command: str) -> List[str]:
        """
        Return docker CMD form of the command
        """

        if self.job.resources.get("aws_batch_unique_workdir", False):
            return ["/bin/bash", "-c", f"mkdir {self.container_unique_workdir} && cd {self.container_unique_workdir} && " + remote_command]
        return ["/bin/bash", "-c", remote_command]

    def _validate_resources(self, vcpu: str, mem: str) -> tuple[str, str]:
        """Validates vcpu and meme conform to Batch EC2 cpu/mem relationship

        https://docs.aws.amazon.com/batch/latest/APIReference/API_ResourceRequirement.html
        """
        vcpu = int(vcpu)
        mem = int(mem)

        if mem in VALID_RESOURCES_MAPPING:
            if vcpu in VALID_RESOURCES_MAPPING[mem]:
                return str(vcpu), str(mem)
            else:
                raise WorkflowError(f"Invalid vCPU value {vcpu} for memory {mem} MB")
        else:
            min_mem = min([m for m, v in VALID_RESOURCES_MAPPING.items() if vcpu in v])
            self.logger.warning(
                f"Memory value {mem} MB is invalid for vCPU {vcpu}."
                f"Setting memory to minimum allowed value {min_mem} MB."
            )
            return str(vcpu), str(min_mem)

    def build_job_definition(self):
        job_uuid = str(uuid.uuid4())
        job_name = f"snakejob-{self.job.name}-{job_uuid}"
        job_definition_name = f"snakejob-def-{self.job.name}-{job_uuid}"

        # Validate and convert resources
        gpu = max(0, int(self.job.resources.get("_gpus", 0)))
        vcpu = max(1, int(self.job.resources.get("_cores", 1)))  # Default to 1 vCPU
        mem = max(1, int(self.job.resources.get("mem_mb", 2048)))  # Default to 2048 MiB

        vcpu_str, mem_str = self._validate_resources(str(vcpu), str(mem))
        gpu_str = str(gpu)

        environment = []
        if self.envvars:
            environment = [{"name": k, "value": v} for k, v in self.envvars.items()]

        container_properties = {
            "image": self.container_image,
            # command requires a list of strings (docker CMD format)
            "command": self._make_container_command(self.job_command),
            "environment": environment,
            "jobRoleArn": self.settings.job_role,
            "privileged": True,
            "resourceRequirements": [
                {
                    "type": BATCH_JOB_RESOURCE_REQUIREMENT_TYPE.VCPU.value,
                    "value": vcpu_str,
                },
                {
                    "type": BATCH_JOB_RESOURCE_REQUIREMENT_TYPE.MEMORY.value,
                    "value": mem_str,
                },
            ],
        }

        if gpu > 0:
            container_properties["resourceRequirements"].append(
                {
                    "type": BATCH_JOB_RESOURCE_REQUIREMENT_TYPE.GPU.value,
                    "value": gpu_str,
                }
            )

        if all((
            self.job.resources.get("volume"),
            self.job.resources.get("fs_id"),
            self.job.resources.get("fsap_id")
        )):
            container_properties.update({
                "mountPoints": [
                    {
                        "sourceVolume": self.job.resources.get("volume"),
                        "containerPath": self.container_path,
                        "readOnly": False
                    }
                ],
                "volumes": [
                    {
                        "name": self.job.resources.get("volume"),
                        "efsVolumeConfiguration": {
                            "fileSystemId": self.job.resources.get("fs_id"),
                            "rootDirectory": "/",
                            "transitEncryption": "ENABLED",
                            "authorizationConfig": {
                                "accessPointId": self.job.resources.get("fsap_id"),
                                "iam": "ENABLED"
                            }
                        }
                    }
                ]
            })

        timeout = {"attemptDurationSeconds": self.settings.task_timeout}
        tags = self.settings.tags if isinstance(self.settings.tags, dict) else dict()
        try:
            job_def = self.batch_client.register_job_definition(
                jobDefinitionName=job_definition_name,
                type=BATCH_JOB_DEFINITION_TYPE.CONTAINER.value,
                containerProperties=container_properties,
                timeout=timeout,
                tags=tags,
                platformCapabilities=[BATCH_JOB_PLATFORM_CAPABILITIES.EC2.value],
            )
            self.created_job_defs.append(job_def)
            return job_def, job_name
        except Exception as e:
            raise WorkflowError(f"Failed to register job definition: {e}") from e

    def submit(self):
        job_def, job_name = self.build_job_definition()

        job_params = {
            "jobName": job_name,
            "jobQueue": self.settings.job_queue,
            "jobDefinition": "{}:{}".format(
                job_def["jobDefinitionName"], job_def["revision"]
            ),
        }

        try:
            submitted = self.batch_client.submit_job(**job_params)
            return submitted
        except Exception as e:
            raise WorkflowError(f"Failed to submit job: {e}") from e
