import io
import pathlib

import numpy as np
import pytest

import meshio

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.polygon_mesh,
    ],
)
def test_obj(mesh, tmp_path):
    for k, c in enumerate(mesh.cells):
        mesh.cells[k] = meshio.CellBlock(c.type, c.data.astype(np.int32))

    helpers.write_read(tmp_path, meshio.obj.write, meshio.obj.read, mesh, 1.0e-12)


@pytest.mark.skip("Fails point data consistency check.")
@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells", [("elephav.obj", 3.678372172450000e05, 1148)]
)
def test_reference_file(filename, ref_sum, ref_num_cells):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "obj" / filename

    mesh = meshio.read(filename)
    tol = 1.0e-5
    s = np.sum(mesh.points)
    assert abs(s - ref_sum) < tol * abs(ref_sum)
    assert mesh.cells[0].type == "triangle"
    assert len(mesh.cells[0].data) == ref_num_cells


@pytest.mark.parametrize("num_vertices", [3, 4, 5])
@pytest.mark.parametrize("references", ["positive", "negative", "mixed"])
@pytest.mark.parametrize("suffix", ["", "/1", "//1", "/1/1"])
@pytest.mark.parametrize("trailing_vertex", [False, True])
def test_face_vertex_references(num_vertices, references, suffix, trailing_vertex):
    # OBJ references are relative to the vertex list at the face, not at EOF.
    lines = [f"v {i} {i % 2} 0" for i in range(num_vertices)]
    if suffix:
        lines += ["vt 0 0", "vn 0 0 1"] * num_vertices
    indices = []
    for i in range(num_vertices):
        index = i + 1
        if references == "negative" or (references == "mixed" and i % 2):
            index = i - num_vertices
        indices.append(f"{index}{suffix}")
    lines.append("f " + " ".join(indices))
    if trailing_vertex:
        lines.append("v 99 99 99")
        if suffix:
            lines += ["vt 0 0", "vn 0 0 1"]

    mesh = meshio.obj.read(io.StringIO("\n".join(lines) + "\n"))

    assert mesh.cells[0].type == {3: "triangle", 4: "quad", 5: "polygon"}[num_vertices]
    np.testing.assert_array_equal(mesh.cells[0].data, [range(num_vertices)])
    np.testing.assert_array_equal(
        mesh.points[mesh.cells[0].data[0]],
        [[i, i % 2, 0] for i in range(num_vertices)],
    )


@pytest.mark.parametrize("file_format", ["obj", "vtu"])
def test_interleaved_relative_faces(tmp_path, file_format):
    text = """v 0 0 0
v 1 0 0
v 0 1 0
g first
f -3 -2 -1
v 0 0 1
f 1 -2 -1
g second
f -4 -3 -2 -1
v 2 2 2
f -5 -4 -3 -2 -1
"""
    mesh = meshio.obj.read(io.StringIO(text))
    expected = [[[0, 1, 2], [0, 2, 3]], [[0, 1, 2, 3]], [[0, 1, 2, 3, 4]]]
    assert [c.type for c in mesh.cells] == ["triangle", "quad", "polygon"]
    for cells, indices in zip(mesh.cells, expected):
        np.testing.assert_array_equal(cells.data, indices)
    for groups, expected_groups in zip(
        mesh.cell_data["obj:group_ids"], [[0, 0], [1], [1]]
    ):
        np.testing.assert_array_equal(groups, expected_groups)

    filename = tmp_path / f"relative.{file_format}"
    meshio.write(filename, mesh, file_format=file_format)
    reread = meshio.read(filename)
    for cells, indices in zip(reread.cells, expected):
        np.testing.assert_array_equal(cells.data, indices)
    np.testing.assert_array_equal(reread.points, mesh.points)
