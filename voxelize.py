import numpy as np
import plotly.graph_objects as go
import struct

def array_to_mesh(arr):
  """
  Create a 3D mesh from a 3D numpy array.

  Args:
    arr (np.ndarray): 3D numpy array.
  Returns:
    Tuple of mesh vertex coordinates and mesh faces.
  """

  if arr.ndim != 3:
    raise ValueError("Input array must be 3D.")

  def occupied(x, y, z):
    return (
      0 <= x < arr.shape[0] and 
      0 <= y < arr.shape[1] and 
      0 <= z < arr.shape[2] and 
      arr[x, y, z]
    )

  cube_vertices = [
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
  ]

  pos_to_index = {}
  vertices = []
  faces = []
  on = np.argwhere(arr)
  for x, y, z in on:
    local_to_global = [0] * 8
    for i, (dx, dy, dz) in enumerate(cube_vertices):
      pos = (x + dx, y + dy, z + dz)
      if pos in pos_to_index:
        local_to_global[i] = pos_to_index[pos]
      else:
        local_to_global[i] = pos_to_index[pos] = len(vertices)
        vertices.append(pos)

    def add_face(i, j, k, l):
      faces.append([local_to_global[i], local_to_global[j], 
                    local_to_global[k], local_to_global[l]])
      # faces.append([local_to_global[i], local_to_global[j], local_to_global[k]])
      # faces.append([local_to_global[k], local_to_global[l], local_to_global[i]])

    if not occupied(x - 1, y, z):
      add_face(0, 3, 7, 4)
    if not occupied(x + 1, y, z):
      add_face(1, 2, 6, 5)
    if not occupied(x, y - 1, z):
      add_face(0, 1, 5, 4)
    if not occupied(x, y + 1, z):
      add_face(2, 3, 7, 6)
    if not occupied(x, y, z - 1):
      add_face(0, 1, 2, 3)
    if not occupied(x, y, z + 1):
      add_face(4, 5, 6, 7)

  return np.array(vertices), np.array(faces)

arr = np.load('pass_winner_76_160.npy')
arr = arr[30:, :, :]
print(f'Voxels: {np.count_nonzero(arr)}')

vertices, faces = array_to_mesh(arr)
print(f'Vertices: {vertices.shape[0]}, Faces: {faces.shape[0]}')
with open('mesh.bin', 'wb') as f:
  f.write(struct.pack('ii', vertices.shape[0], faces.shape[0]))
  for vertex in vertices:
    f.write(struct.pack('iii', *vertex))
  for face in faces:
    f.write(struct.pack('iiii', *face))

# fig = go.Figure(data=[
#   go.Mesh3d(
#     x=vertices[:, 0],
#     y=vertices[:, 1],
#     z=vertices[:, 2],
#     i=faces[:, 0],
#     j=faces[:, 1],
#     k=faces[:, 2],
#     flatshading=True,
#     lighting=dict(
#       ambient=0.5, 
#       diffuse=0.5, 
#       fresnel=4, 
#       specular=0.2, 
#       roughness=0.8, 
#       facenormalsepsilon=0
#     ),
#     lightposition=dict(x=1000, y=1000, z=0),
#     colorscale='mint',
#     intensity=vertices[:, 0]
#   )
# ])

# fig.update_layout(
#   scene=dict(
#     xaxis=dict(visible=False),
#     yaxis=dict(visible=False),
#     zaxis=dict(visible=False),
#     aspectmode='data',
#   ),
# )
# fig.update_traces(showscale=False)

# # fig.show()
# fig.write_html('voxel.html', include_plotlyjs='cdn', full_html=False)