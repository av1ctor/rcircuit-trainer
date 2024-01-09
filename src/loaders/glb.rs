use gltf::Gltf;
use three_rs::math::{matrix4::Matrix4, triangle::Triangle, vector3::Vector3};

pub fn load_from_file(
    path: &str
) -> Result<Vec<Triangle>, String> {
    let bytes = std::fs::read(path).map_err(|err| err.to_string())?;
    load_from_bytes(&bytes)
}

pub fn load_from_bytes(
    bytes: &[u8]
) -> Result<Vec<Triangle>, String> {
    let gltf = Gltf::from_slice(&bytes).unwrap();
    let buffers = gltf::import_buffers(&gltf.document, None, gltf.blob).unwrap();
    load(&gltf.document.default_scene().unwrap(), &buffers)
}

pub fn load<'a>(
    scene: &gltf::Scene<'_>,
    buffers: &Vec<gltf::buffer::Data>
) -> Result<Vec<Triangle>, String> {
    
    let mut triangles = vec![];
    
    for node in scene.nodes() {
        traverse_meshes(
            &node,
            None, 
            &mut |mesh, world_matrix| {
                for primitive in mesh.primitives() {
                    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                            
                    if let Some(iter) = reader.read_positions() {
                        let vertices = iter.collect::<Vec<_>>();
                        let indices: Vec<u32> = reader.read_indices()
                            .map(|ind| ind.into_u32().collect())
                            .unwrap_or((0..vertices.len() as u32).collect());

                        let mut tris = vec![];
                
                        match primitive.mode() {
                            gltf::mesh::Mode::Triangles => {
                                for i in (0..indices.len()).step_by(3) {
                                    tris.push([
                                        vertices[indices[i] as usize],
                                        vertices[indices[i + 1] as usize],
                                        vertices[indices[i + 2] as usize],
                                    ]);
                                }
                            }
                            gltf::mesh::Mode::TriangleStrip => {
                                for i in 0..(indices.len() - 2) {
                                    tris.push([
                                        vertices[indices[i] as usize + i % 2],
                                        vertices[indices[i + 1 - i % 2] as usize],
                                        vertices[indices[i + 2] as usize],
                                    ]);
                                }
                            }
                            gltf::mesh::Mode::TriangleFan => {
                                for i in 1..(indices.len() - 1) {
                                    tris.push([
                                        vertices[indices[0] as usize],
                                        vertices[indices[i] as usize],
                                        vertices[indices[i + 1] as usize],
                                    ]);
                                }
                            }
                            _ => {
                                return Err("Unsupported primitive".to_string());
                            },
                        }

                        for tri in tris {
                            let a = Vector3::from_slice(&tri[0]).apply_matrix4(world_matrix);
                            let b = Vector3::from_slice(&tri[1]).apply_matrix4(world_matrix);
                            let c = Vector3::from_slice(&tri[2]).apply_matrix4(world_matrix);

                            triangles.push(Triangle::new(a, b, c));
                        }
                    }
                }

                Ok(())
            }
        )?;
    }

    Ok(triangles)
}

fn traverse_meshes(
    node: &gltf::Node<'_>,
    world_matrix: Option<&Matrix4>,
    cb: &mut dyn FnMut (&gltf::Mesh<'_>, &Matrix4) -> Result<(), String>
) -> Result<(), String> {
    
    let matrix = Matrix4::from_slice2(&node.transform().matrix());
    let world_matrix = if let Some(m) = world_matrix {
        m.mul(&matrix)
    }
    else {
        matrix
    };
    
    if let Some(mesh) = node.mesh() {
        cb(&mesh, &world_matrix)?;
        for child in node.children() {
            traverse_meshes(&child, Some(&world_matrix), cb)?;
        }
    }
    else {
        for child in node.children() {
            traverse_meshes(&child, Some(&world_matrix), cb)?;
        }
    }

    Ok(())
}