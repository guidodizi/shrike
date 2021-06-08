# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Unit tests for the `prepare` module.

Consider decorating long end-to-end tests with `@pytest.mark.order(-1)`.
"""

import pytest
import os
from pathlib import Path
import subprocess
import shutil
import yaml
from git import Repo

from shrike.build.commands import prepare
from shrike.build.core.configuration import Configuration


def clean() -> None:
    """
    Clean up all non checked-in files.
    """
    subprocess.run(["git", "clean", "-xdf"])


@pytest.mark.order(-1)
def test_build_all_components(caplog):
    prep = prepare.Prepare()
    prep.config = Configuration(signing_mode="foo")
    prep.ensure_component_cli_installed()
    component = "tests/tests_build/steps/component1/spec.yaml"
    with caplog.at_level("INFO"):
        success = prep.build_all_components([component])
    assert success
    assert f"Component {component} is built." in caplog.text


@pytest.mark.parametrize("mode", ["foo", "aether", "aml"])
def test_find_component_specification_files_using_all(mode):
    # clean the .build directories first, such that we won't include the
    # spec.yaml files under .build directories in the unit tests here
    clean()

    prep = prepare.Prepare()
    prep.config = Configuration(signing_mode=mode)

    dir = str(Path(__file__).parent.parent.resolve() / "steps/component1")
    print(dir)
    res = prep.find_component_specification_files_using_all(dir=dir)
    print("found spec yaml file paths list: ", res)
    assert len(res) == 1

    dir = str(Path(__file__).parent.parent / "steps/component1")
    print(dir)
    res = prep.find_component_specification_files_using_all(dir=dir)
    print("found spec yaml file paths list: ", res)
    assert len(res) == 1

    dir = str(Path(__file__).parent.parent.resolve() / "steps/component2")
    print(dir)
    res = prep.find_component_specification_files_using_all(dir=dir)
    print("found spec yaml file paths list: ", res)
    assert len(res) == 1

    dir = str(Path(__file__).parent.parent / "steps/component2")
    print(dir)
    res = prep.find_component_specification_files_using_all(dir=dir)
    print("found spec yaml file paths list: ", res)
    assert len(res) == 1

    dir = str(Path(__file__).parent.parent.resolve() / "steps/component3")
    print(dir)
    res = prep.find_component_specification_files_using_all(dir=dir)
    print("found spec yaml file paths list: ", res)
    assert len(res) == 1

    dir = str(Path(__file__).parent.parent / "steps/component3")
    print(dir)
    res = prep.find_component_specification_files_using_all(dir=dir)
    print("found spec yaml file paths list: ", res)
    assert len(res) == 1


def test_create_catalog_files_fails_if_non_standard_mode():
    with pytest.raises(ValueError):
        prep = prepare.Prepare()
        prep.config = Configuration(signing_mode="foo")
        prep.create_catalog_files([""])


def test_create_catalog_files_fails_if_non_standard():
    with pytest.raises(ValueError):
        prep = prepare.Prepare()
        prep.config = Configuration(signing_mode="foo")
        prep.create_catalog_files(
            [
                "tests/tests_build/steps/component1/spec.yaml",
                "tests/tests_build/steps/component2/spec.yaml",
            ]
        )


def test_non_standard_activation_method_not_supported():
    with pytest.raises(ValueError):
        prep = prepare.Prepare()
        prep.config = Configuration(activation_method="foo")
        prep.find_component_specification_files()


def test_smart_method_is_supported():
    prep = prepare.Prepare()
    prep.config = Configuration(activation_method="smart")
    prep.find_component_specification_files()


def test_create_catalog_files_for_aether(caplog):
    prep = prepare.Prepare()
    prep.config = Configuration(signing_mode="aether")
    prep.ensure_component_cli_installed()
    component = [
        "tests/tests_build/steps/component1/spec.yaml",
        "tests/tests_build/steps/component2/spec.yaml",
    ]
    with caplog.at_level("INFO"):
        prep.create_catalog_files_for_aether(component)
    assert "Finish creating aether catalog files for component1." in caplog.text
    assert "Finish creating aether catalog files for component2." in caplog.text
    assert os.path.exists("tests/tests_build/steps/component1/component1.cat")
    assert os.path.exists("tests/tests_build/steps/component2/component2.cat")
    os.remove("tests/tests_build/steps/component1/component1.cat")
    os.remove("tests/tests_build/steps/component2/component2.cat")


@pytest.mark.parametrize(
    "modified_file,component",
    [
        (
            "./tests/tests_build/steps/component1/another_file_for_component1.txt",
            "./tests/tests_build/steps/component1/spec.yaml",
        ),
        (
            "./tests/tests_build/steps/component2/subdir1/subsubdir1/file_in_subsubdir1.txt",
            "./tests/tests_build/steps/component2/spec.yaml",
        ),
        (
            "./tests/tests_build/steps/component2/subdir2/file_in_subdir2.txt",
            "./tests/tests_build/steps/component2/spec.yaml",
        ),
        (
            "./tests/tests_build/steps/component2/subdir2/some_deleted_file.txt",
            "./tests/tests_build/steps/component2/spec.yaml",
        ),
        (
            "./tests/tests_build/steps/component2/some_deleted_subdir/some_deleted_file.txt",
            "./tests/tests_build/steps/component2/spec.yaml",
        ),
    ],
)
def test_is_in_subfolder(modified_file, component):
    prep = prepare.Prepare()
    res = prep.is_in_subfolder(modified_file, component)
    assert res


@pytest.mark.parametrize(
    "modified_file,component",
    [
        (
            "./tests/tests_build/steps/component1/another_file_for_component1.txt",
            "./tests/tests_build/steps/component2/spec.yaml",
        ),
        (
            "./tests/tests_build/steps/component2/subdir1/subsubdir1/file_in_subsubdir1.txt",
            "./tests/tests_build/steps/component1/spec.yaml",
        ),
        (
            "./tests/tests_build/commands/test_prepare.py",
            "./tests/tests_build/steps/component1/spec.yaml",
        ),
        (
            "./tests/tests_build/steps/component1/some_deleted_file.txt",
            "./tests/tests_build/steps/component2/spec.yaml",
        ),
        (
            "./tests/tests_build/steps/deleted_component/subdir/some_deleted_file.txt",
            "./tests/tests_build/steps/component2/spec.yaml",
        ),
        (
            "./tests/tests_build/steps/component2/subdir2/file_in_subdir2.txt",
            "./tests/tests_build/steps/deleted_component/spec.yaml",
        ),
    ],
)
def test_is_not_in_subfolder(modified_file, component):
    prep = prepare.Prepare()
    res = prep.is_in_subfolder(modified_file, component)
    assert res == False


@pytest.mark.parametrize(
    "modified_file,component_additional_includes_contents",
    [
        (
            "./shrike/build/commands/prepare.py",
            [
                "./shrike/build/commands",
                "./shrike/build/non_existent_directory",
            ],
        ),
        (
            "./shrike/build/commands/some_deleted_file.txt",
            [
                "./shrike/build/commands",
                "./shrike/build/non_existent_directory",
            ],
        ),
        (
            "./shrike/build/commands/prepare.py",
            [
                "./shrike/build/commands/prepare.py",
                "./shrike/build/non_existent_directory",
            ],
        ),
        (
            "./shrike/build/commands/prepare.py",
            ["./shrike/build", "./shrike/build/non_existent_directory"],
        ),
    ],
)
def test_is_in_additional_includes(
    modified_file, component_additional_includes_contents
):
    prep = prepare.Prepare()
    res = prep.is_in_additional_includes(
        modified_file, component_additional_includes_contents
    )
    assert res


@pytest.mark.parametrize(
    "modified_file,component_additional_includes_contents",
    [
        (
            "./shrike/build/commands/prepare.py",
            [
                "./shrike/build/core",
                "./shrike/build/tests",
                "./shrike/build/non_existent_directory",
            ],
        ),
        (
            "./shrike/build/commands/some_deleted_file.txt",
            [
                "./shrike/build/core",
                "./shrike/build/tests",
                "./shrike/build/non_existent_directory",
                "./shrike/build/non_existent_directory/non_existent_subdirectory",
            ],
        ),
        (
            "./shrike/build/commands/some_deleted_directory/some_deleted_file.txt",
            [
                "./shrike/build/core",
                "./shrike/build/tests",
                "./shrike/build/non_existent_directory",
            ],
        ),
        (
            "./shrike/build/commands/some_deleted_directory/some_deleted_subsirectory/some_deleted_file.txt",
            [
                "./shrike/build/core",
                "./shrike/build/tests",
                "./shrike/build/non_existent_directory",
            ],
        ),
        (
            "./shrike/build/__init__.py",
            [
                "./shrike/build/core",
                "./shrike/build/tests",
                "./shrike/build/non_existent_directory",
            ],
        ),
    ],
)
def test_is_not_in_additional_includes(
    modified_file, component_additional_includes_contents
):
    prep = prepare.Prepare()
    res = prep.is_in_additional_includes(
        modified_file, component_additional_includes_contents
    )
    assert res == False


@pytest.mark.parametrize(
    "component,modified_files",
    [
        (
            "./tests/tests_build/steps/component1/spec.yaml",
            [
                "./shrike/build/commands/prepare.py",
            ],
        ),
        (
            "./tests/tests_build/steps/component1/spec.yaml",
            [
                "./tests/tests_build/commands/test_prepare.py",
            ],
        ),
        (
            "./tests/tests_build/steps/component2/spec.yaml",
            [
                "./tests/tests_build/steps/component2/subdir1/subsubdir1/file_in_subsubdir1.txt",
            ],
        ),
        (
            "./tests/tests_build/steps/component2/spec.yaml",
            [
                "./tests/tests_build/steps/component2/subdir2/file_in_subsubdir2.txt",
            ],
        ),
        (
            "./tests/tests_build/steps/component3/spec.yaml",
            [
                "./tests/tests_build/steps/component3/spec.yaml",
            ],
        ),
        (
            "./tests/tests_build/steps/component3/spec.yaml",
            [
                "./tests/tests_build/steps/component3/some_deleted_file.txt",
            ],
        ),
    ],
)
def test_component_is_active(component, modified_files):
    prep = prepare.Prepare()
    res = prep.component_is_active(component, modified_files)
    assert res


@pytest.mark.parametrize(
    "component,modified_files",
    [
        (
            "./tests/tests_build/steps/deleted_component/spec.yaml",
            [
                "./shrike/build/commands/prepare.py",
            ],
        ),
        (
            "./tests/tests_build/steps/component1/spec.yaml",
            [
                "./tests/tests_build/steps/deleted_component/deleted_file.txt",
            ],
        ),
    ],
)
def test_component_is_active_for_deleted_component_or_file(component, modified_files):
    prep = prepare.Prepare()
    res = prep.component_is_active(component, modified_files)
    assert res == False


@pytest.mark.parametrize(
    "component,modified_files",
    [
        (
            "./tests/tests_build/steps/component2/spec.yaml",
            [
                "./shrike/build/commands/prepare.py",
            ],
        ),
        (
            "./tests/tests_build/steps/component2/spec.yaml",
            [
                "./tests/tests_build/commands/test_prepare.py",
            ],
        ),
        (
            "./tests/tests_build/steps/component2/spec.yaml",
            [
                "./tests/tests_build/steps/component1/spec.yaml",
            ],
        ),
    ],
)
def test_component_is_not_active(component, modified_files):
    prep = prepare.Prepare()
    res = prep.component_is_active(component, modified_files)
    assert res == False


@pytest.mark.parametrize(
    "modified_files, expected_res",
    [
        (
            [
                "./shrike/build/commands/prepare.py",
            ],
            [
                "./tests/tests_build/steps/component1/spec.yaml",
            ],
        ),
        (
            [
                "./tests/tests_build/commands/test_prepare.py",
            ],
            [
                "./tests/tests_build/steps/component1/spec.yaml",
            ],
        ),
        (
            [
                "./tests/tests_build/steps/component2/subdir1/subsubdir1/file_in_subsubdir1.txt",
            ],
            [
                "./tests/tests_build/steps/component2/spec.yaml",
            ],
        ),
        (
            [
                "./tests/tests_build/steps/component2/subdir2/file_in_subdir2.txt",
                "./tests/tests_build/steps/component3/spec.yaml",
            ],
            [
                "./tests/tests_build/steps/component2/spec.yaml",
                "./tests/tests_build/steps/component3/spec.yaml",
            ],
        ),
    ],
)
def test_infer_active_components_from_modified_files(modified_files, expected_res):
    prep = prepare.Prepare()
    prep.config = Configuration()
    res = prep.infer_active_components_from_modified_files(modified_files)
    for res_line_number in range(0, len(res)):
        assert res[res_line_number] == str(
            Path(expected_res[res_line_number]).resolve()
        )
    assert len(res) == len(expected_res)


def test_get_modified_files():

    # Creating a new repo and declaring branch names
    main_branch_name = "main"
    new_branch_name = "newbranch"
    root_repo_location = "./temp_repo/"
    if Path(root_repo_location).exists():
        shutil.rmtree(root_repo_location)
    repo_path = Path(root_repo_location) / "bare-repo"
    print("New repo path: " + str(repo_path.resolve()))
    new_repo = Repo.init(repo_path, initial_branch="main")

    # First commit to main
    # creating some files
    file_name_1 = repo_path / "new-file-1.txt"
    open(file_name_1, "wb").close()
    file_name_2 = repo_path / "new-file-2.py"
    open(file_name_2, "wb").close()
    subdir_path = repo_path / "subdirectory"
    try:
        os.mkdir(subdir_path)
    except:
        print(str(subdir_path) + " already exists; no need to create it.")
    file_name_3 = subdir_path / "new-file-3.yaml"
    open(file_name_3, "wb").close()
    # add them to the index
    new_repo.index.add(
        [
            str(file_name_1.resolve()),
            str(file_name_2.resolve()),
            str(file_name_3.resolve()),
        ]
    )  #
    # do the commit
    new_repo.index.commit("Merged PR: First one")

    # Second commit to main
    # create a new file
    file_name_4 = repo_path / "new-file-4.py"
    open(file_name_4, "wb").close()
    # modify an old file
    with open(file_name_1, "w") as file:
        file.write("This is a change to the first file\n")
    new_repo.index.add([str(file_name_1.resolve()), str(file_name_4.resolve())])
    new_repo.index.commit("Merged PR: Second one")

    # create a "remote" and push everything to it (a remote is needed for the RB case)
    remote_repo_path_string = "../remote-repo"  # relative to 'repo_path'
    print("Remote repo path: " + str(remote_repo_path_string))
    cloned_repo = new_repo.clone(remote_repo_path_string)
    new_repo.create_remote("origin", url=remote_repo_path_string)
    new_repo.remotes.origin.pull(refspec=main_branch_name + ":origin")

    # Third commit, to a different branch and with a file deletion
    file_name_5 = repo_path / "new-file-5.py"
    open(file_name_5, "wb").close()
    # modify an old file
    with open(file_name_2, "w") as file:
        file.write("This is a change to the second file\n")
    # delete a file
    os.remove(file_name_3.resolve())
    new_repo.git.checkout("HEAD", b=new_branch_name)
    new_repo.index.add([str(file_name_2.resolve()), str(file_name_5.resolve())])
    new_repo.index.remove([str(file_name_3.resolve())])
    new_repo.index.commit("This is a commit in the non-compliant branch")

    # now we're ready to do the actual tests
    prep = prepare.Prepare()
    prep.config = Configuration()
    # 1. test the 'Build - after Merge' case (BAM)
    change_list_BAM = prep.get_modified_files(
        new_repo, main_branch_name, main_branch_name
    )
    assert change_list_BAM == {
        str(file_name_1.resolve()),
        str(file_name_4.resolve()),
    }
    # 2. test the 'Build - before Merge' case (BBM)
    change_list_BBM = prep.get_modified_files(
        new_repo, "refs/pull/XXXXXX/merge", main_branch_name
    )
    assert change_list_BBM == {
        str(file_name_2.resolve()),
        str(file_name_3.resolve()),
        str(file_name_5.resolve()),
    }
    # 3. test the 'Manual' case (triggered from local)
    change_list_Manual = prep.get_modified_files(
        new_repo, new_branch_name, main_branch_name
    )
    assert change_list_Manual == {
        str(file_name_2.resolve()),
        str(file_name_3.resolve()),
        str(file_name_5.resolve()),
    }

    # 4. test the 'Manual' case triggered from DevOps
    # (using a non existing branch name to mimick that DevOps case where the repo has detached head and we cannot just take repo.heads[current_branch])
    change_list_Manual = prep.get_modified_files(
        new_repo, "non-existing_branch_name", main_branch_name
    )
    assert change_list_Manual == {
        str(file_name_2.resolve()),
        str(file_name_3.resolve()),
        str(file_name_5.resolve()),
    }

    # clean up the newly created repos
    new_repo.close()
    cloned_repo.close()
    shutil.rmtree(root_repo_location)


def test_identify_repo_and_branches():
    prep = prepare.Prepare()
    prep.config = Configuration(compliant_branch="^refs/heads/CompliantBranchName$")
    [repo, current_branch, compliant_branch] = prep.identify_repo_and_branches()
    assert repo.bare == False
    assert current_branch
    assert compliant_branch == "CompliantBranchName"


def test_identify_repo_and_branches_when_no_repo():
    prep = prepare.Prepare()
    prep.config = Configuration(
        compliant_branch="^refs/heads/CompliantBranchName$",
        working_directory=str(Path(__file__).parents[4]),
    )  # we set the working directory above the repo level, so there's no repo to be found
    print("Working directory: " + str(prep.config.working_directory))
    with pytest.raises(Exception):
        [repo, current_branch, compliant_branch] = prep.identify_repo_and_branches()


@pytest.mark.order(-1)
def test_ensure_component_cli_installed(caplog):
    prep = prepare.Prepare()
    prep.config = Configuration(activation_method="foo")
    with caplog.at_level("INFO"):
        first_try = prep.ensure_component_cli_installed()
        second_try = prep.ensure_component_cli_installed()

    assert second_try == True
    assert "component CLI exists. Skipping installation." in caplog.text


@pytest.mark.order(-1)
def test_log_info_component_cli_installed(caplog):
    prep = prepare.Prepare()
    prep.config = Configuration(activation_method="foo")
    with caplog.at_level("ERROR"):
        prep.ensure_component_cli_installed()
    assert "Command failed with exit code" not in caplog.text


@pytest.mark.parametrize("catalog_file_name", ["catalog.json", "catalog.json.sig"])
def test_create_catalog_files_for_aml(catalog_file_name):
    prep = prepare.Prepare()
    prep.config = Configuration(suppress_adding_repo_pr_tags=True)
    # we'll test 2 component folder structures
    component1_folder = "tests/tests_build/steps/component1/"  # flat directory
    component2_folder = "tests/tests_build/steps/component2/"  # nested directories
    # create the catalogs
    prep.create_catalog_files_for_aml(
        [
            os.path.join(component1_folder, "spec.yaml"),
            os.path.join(component2_folder, "spec.yaml"),
        ]
    )
    # grab the freshly created catalogs
    component1_catalog_path = os.path.join(component1_folder, catalog_file_name)
    with open(component1_catalog_path, "r") as component1_catalog_file:
        component1_catalog_contents = component1_catalog_file.read()
    component2_catalog_path = os.path.join(component2_folder, catalog_file_name)
    with open(component2_catalog_path, "r") as component2_catalog_file:
        component2_catalog_contents = component2_catalog_file.read()
    # hard-coded values to check
    component1_catalog_reference = '{"HashAlgorithm": "SHA256", "CatalogItems": {"another_file_for_component1.txt": "9E640A18FC586A6D87716F9B4C6728B7023819E58E07C4562E0D2C14DFC3CF5B", "spec.additional_includes": "50407DAA1E6DA1D91E1CE88DDDF18B3DFDA62E08B780EC9B2E8642536DD36C05", "spec.yaml": "BF41A2F4D427A281C0E7EB5F987E285D69F5D50AEFCBB559DC2D7611D861D7FA"}}'
    # component2_catalog_reference = '{"HashAlgorithm": "SHA256", "CatalogItems": {".amlignore": "1CA13EBDBB24D532673E22A0886631976CDC9B9A94488FE31AF9214F4A79E8AE", ".subdir/.gitignore": "A5270F91138FC2BB5470ECB521DAB043140D7E0FD8CB33BB0644AC13EFB60FE7", "another_file_for_component2.txt": "A47275ACA3BC482FC3F4C922572EA514D0DE03EA836597D34FC21BA805D2ABCA", "spec.yaml": "B1CBC48FFB11EBC9C3C84CD0F6BF852EF9573C3C34E83DCD165FF66074B19DFF", "subdir1/.gitignore": "C0159813AB6EAF3CC8A0BD37C79E4CDD927E3E95CB9BA8EC246BC3A176C3EB41", "subdir1/file_in_subdir1.txt": "A7917FCCF0C714716F308967DB45B2DDEE4665FC4B4FCC6C0E50ABD55DD1C6B5", "subdir1/subsubdir1/file_in_subsubdir1.txt": "1791DD6583A06429603CC30CDC2AE6A217853722C6BB10AA31027F5A931D5A7D", "subdir2/file_in_subdir2.txt": "419EE822D1E34B22FCE7F09EDCFC7565188A1362352E1DADB569820CB599D651"}}'
    # assertions
    assert component1_catalog_contents == component1_catalog_reference
    # Note: The hash values of component2 always change after we have a big PR.
    # I tentatively commented it, but may uncomment it in the future.
    # assert component2_catalog_contents == component2_catalog_reference


def test_validate_all_components_does_nothing_if_no_files(caplog):
    prep = prepare.Prepare()

    with caplog.at_level("INFO"):
        prep.validate_all_components([])

    assert not caplog.text


@pytest.mark.order(-1)
def test_validate_all_components_works_on_invalid_component():
    component_path = str(
        Path(__file__).parent.parent.resolve() / "steps/component1/spec.yaml"
    )

    prep = prepare.Prepare()
    prep.config = Configuration()

    prep.attach_workspace(
        "/subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourceGroups/aml1p-rg/providers/Microsoft.MachineLearningServices/workspaces/aml1p-ml-eus2"
    )
    success = prep.validate_all_components([component_path])

    assert not success
    assert prep._component_statuses[component_path]["validate"] == "failed"
    assert len(prep._errors) == 1


@pytest.mark.order(-1)
def test_validate_all_components_works_on_valid_component(caplog):
    component_path = str(
        Path(__file__).parent.parent.resolve() / "steps/component2/spec.yaml"
    )

    prep = prepare.Prepare()
    prep.config = Configuration()

    prep.attach_workspace(
        "/subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourceGroups/aml1p-rg/providers/Microsoft.MachineLearningServices/workspaces/aml1p-ml-eus2"
    )

    with caplog.at_level("INFO"):
        prep.validate_all_components([component_path])

    assert prep._component_statuses[component_path]["validate"] == "succeeded"
    assert not prep._errors
    assert "is valid" in caplog.text


@pytest.mark.order(-1)
def test_validate_all_components_code_snapshot_parameter(caplog):
    tmp_dir = str(Path(__file__).parent.parent.resolve() / "steps/tmp_dir")
    tmp_yaml = {
        "$schema": "http://azureml/sdk-2-0/CommandComponent.json",
        "name": "dummy_component",
        "version": "0.0.1",
        "type": "CommandComponent",
        "command": "pip freeze",
        "code": "../../",
    }
    os.mkdir(tmp_dir)
    with open(tmp_dir + "/spec.yaml", "w") as tmp_spec:
        yaml.dump(tmp_yaml, tmp_spec)

    prep = prepare.Prepare()
    prep.config = Configuration()

    prep.attach_workspace(
        "/subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourceGroups/aml1p-rg/providers/Microsoft.MachineLearningServices/workspaces/aml1p-ml-eus2"
    )

    with caplog.at_level("INFO"):
        prep.validate_all_components([tmp_dir + "/spec.yaml"])

    assert prep._component_statuses[tmp_dir + "/spec.yaml"]["validate"] == "failed"
    assert len(prep._errors) == 1
    assert (
        "Code snapshot parameter is not supported. Please use .additional_includes for your component."
        in caplog.text
    )

    # Clean up tmp directory
    shutil.rmtree(tmp_dir)


@pytest.mark.order(-1)
def test_validate_all_components_code_Section(caplog):
    component_path = str(
        Path(__file__).parent.parent.resolve() / "steps/component4/spec.yaml"
    )

    prep = prepare.Prepare()
    prep.config = Configuration()

    prep.attach_workspace(
        "/subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourceGroups/aml1p-rg/providers/Microsoft.MachineLearningServices/workspaces/aml1p-ml-eus2"
    )

    with caplog.at_level("INFO"):
        prep.validate_all_components([component_path])

    assert prep._component_statuses[component_path]["validate"] == "succeeded"
    assert not prep._errors
    assert "is valid" in caplog.text
    assert (
        "Code snapshot parameter is not supported. Please use .additional_includes for your component."
        not in caplog.text
    )


@pytest.mark.order(-1)
def test_workspace_attachment_for_invalid_workspace(caplog):
    prep = prepare.Prepare()
    prep.config = Configuration()

    # try to attach a fake (non-ecxisting) workspace. expected to receive a register_error message.
    fake_workspace_id = "/subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourceGroups/aml1p-rg/providers/Microsoft.MachineLearningServices/workspaces/aml1p-ml-canary-fake"
    with caplog.at_level("INFO"):
        prep.attach_workspace(fake_workspace_id)

    assert f"Error!! Failed to attach to {fake_workspace_id}!" in prep._errors
    assert f"Error!! Failed to attach to {fake_workspace_id}!" in caplog.text


@pytest.mark.order(-1)
def test_run_with_config_runs_end_to_end(caplog):
    # clean the .build directories first, such that we won't include the
    # spec.yaml files under .build directories in the unit tests here
    clean()

    prep = prepare.Prepare()
    prep.config = Configuration(
        workspaces=[
            "/subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourceGroups/aml1p-rg/providers/Microsoft.MachineLearningServices/workspaces/aml1p-ml-eus2"
        ]
    )

    with caplog.at_level("INFO"):
        prep.run_with_config()

    assert "Running component preparation logic" in caplog.text


@pytest.mark.parametrize(
    "component,expected_len", [("component1", 3), ("component2", 8), ("component3", 1)]
)
def test_all_files_in_snapshot(component, expected_len):
    clean()

    directory = str(Path(__file__).parent.parent.resolve() / f"steps/{component}")
    prep = prepare.Prepare()
    result = prep.all_files_in_snapshot(f"{directory}/spec.yaml")
    assert len(result) == expected_len


@pytest.mark.order(-1)
def test_add_repo_and_last_pr_to_tags():
    component_path = str(
        Path(__file__).parent.parent.resolve() / "steps/component4/spec.yaml"
    )
    prep = prepare.Prepare()
    prep.config = Configuration()
    prep.attach_workspace(
        "/subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourceGroups/aml1p-rg/providers/Microsoft.MachineLearningServices/workspaces/aml1p-ml-eus2"
    )
    prep.add_repo_and_last_pr_to_tags([component_path])
    with open(component_path) as f:
        spec_file = yaml.load(f)
    spec_tags = spec_file.get("tags")
    assert "repo" in spec_tags
    assert "last_commit_id" in spec_tags

    assert "last_commit_message" in spec_tags
    assert "path_to_component" in spec_tags
    assert "[link to commit]" in spec_file.get("description")
