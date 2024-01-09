use gltf::{Gltf, mesh::Mode};
use three_rs::math::{matrix4::Matrix4, vector3::Vector3};

pub struct Primitive {
    pub mode: Mode,
    pub vertices: Vec<Vector3>,
}

impl Primitive {
    pub fn new(
        mode: Mode,
        vertices: Vec<Vector3>
    ) -> Self {
        Self {
            mode,
            vertices,
        }
    }
}

pub fn load_from_file(
    path: &str
) -> Result<Vec<Primitive>, String> {
    let bytes = std::fs::read(path).map_err(|err| err.to_string())?;
    load_from_bytes(&bytes)
}

pub fn load_from_bytes(
    bytes: &[u8]
) -> Result<Vec<Primitive>, String> {
    let gltf = Gltf::from_slice(&bytes).unwrap();
    let buffers = gltf::import_buffers(&gltf.document, None, gltf.blob).unwrap();
    load(&gltf.document.default_scene().unwrap(), &buffers)
}

pub fn load<'a>(
    scene: &gltf::Scene<'_>,
    buffers: &Vec<gltf::buffer::Data>
) -> Result<Vec<Primitive>, String> {
    
    let mut res = vec![];
    
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

                        let mut prim_vertices = vec![];
                
                        match primitive.mode() {
                            gltf::mesh::Mode::Triangles |
                            gltf::mesh::Mode::TriangleStrip |
                            gltf::mesh::Mode::TriangleFan => {
                                for i in 0..indices.len() {
                                    prim_vertices.push(vertices[indices[i] as usize]);
                                }
                            }
                            _ => {
                                return Err("Unsupported primitive".to_string());
                            },
                        }

                        let mut prims = vec![];
                        for v in &mut prim_vertices {
                            prims.push(Vector3::from_slice(&v).apply_matrix4(world_matrix));
                        }

                        res.push(Primitive{
                            mode: primitive.mode().clone(),
                            vertices: prims,
                        })
                    }
                }

                Ok(())
            }
        )?;
    }

    Ok(res)
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