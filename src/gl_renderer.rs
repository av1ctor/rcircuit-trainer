use std::{cell::RefCell, fs, rc::Rc};
use brains_engine::{
    asset::GameAssetType, 
    env::{Asset, AssetRef, AssetValue}, 
    object::ObjectState, 
    renderer::{ModelRef, Renderer}
};
use three_rs::{
    camera::{PerspectiveCamera, ObjectCamera}, 
    core::{Transformable, Geometrical, Object3d, RGB}, 
    geometry::{Box3, Cone}, 
    loader::Gltf, 
    math::{Vector3, Quaternion}, 
    object::Mesh, 
    renderer::Event, 
    scene::Scene
};

pub struct GlRenderer {
    pub w: usize,
    pub h: usize,
    pub assets: Vec<Rc<RefCell<Mesh>>>,
    pub objects: Vec<Rc<RefCell<Mesh>>>,
    pub scene: Scene,
    pub camera: PerspectiveCamera,
    pub renderer: Rc<RefCell<three_rs::renderer::GlRenderer>>,
}

impl GlRenderer {
    pub fn new(
        w: u32,
        h: u32
    ) -> Result<Self, String> {
        unsafe {
            let renderer = Rc::new(RefCell::new(
                three_rs::renderer::GlRenderer::new("trainer", w, h)
            ));
            let mut camera = PerspectiveCamera::new(
                75.0, 
                (w as f32) / (h as f32), 
                0.1, 
                1000.0
            );
            let scene = Scene::new(renderer.clone());

            camera.rotate_x(-0.5)
                .translate_z(50.0);

            Ok(Self{
                w: w as _,
                h: h as _,
                objects: vec![],
                scene,
                camera,
                renderer,
                assets: vec![],
            })
        }
    }

    pub fn poll_events(
        &mut self
    ) -> Vec<Event> {
        self.renderer.borrow_mut().poll_events()
    }
}

impl Renderer for GlRenderer {
    fn reset(
        &mut self
    ) {
        self.objects = vec![];
        self.scene.reset();
    }

    fn load_asset(
        &mut self,
        asset: &Asset
    ) -> Result<(), String> {
        match GameAssetType::from(asset.typ) {
            GameAssetType::Level => {
                match asset.data.0.get("path").unwrap() {
                    AssetValue::Text(path) => {
                        if let Ok(bytes) = fs::read(path) {
                            let mesh = Gltf::load_from_bytes(&bytes)?;
                            self.assets.push(mesh);
                            Ok(())
                        }
                        else {
                            Err("Path not found".to_string())
                        }
                    },
                    _ => {
                        return Err("Path not found".to_string());
                    }
                }
            },
            GameAssetType::Player => {
                let mesh = Mesh::new(
                    &Cone::new(0.6, 1.8)
                );
                self.assets.push(mesh);
    
                Ok(())
            },
            GameAssetType::Brain => {
                let mesh = Mesh::new(
                    &Box3::new(1.0, 1.0, 1.0)
                );
                self.assets.push(mesh);
    
                Ok(())
            },
            GameAssetType::Exit => {
                let mesh = Mesh::new(
                    &Box3::new(2.0, 2.0, 2.0)
                );
                self.assets.push(mesh);
    
                Ok(())
            },
        }
    }
    
    fn add_model(
        &mut self,
        asset_ref: AssetRef
    ) -> Result<ModelRef, String> {
        if asset_ref >= self.assets.len() {
            return Err(format!("Asset not found: {}", asset_ref));
        }
        
        let mesh = &self.assets[asset_ref];
        let rf = self.objects.len();
        let cloned = Rc::new(RefCell::new(mesh.borrow().clone()));
        self.objects.push(cloned.clone());
        self.scene.add(cloned);
        Ok(rf)
    }

    fn add_from_linestrips(
        &mut self,
        color: RGB,
        lines: Vec<Vector3>
    ) -> Result<ModelRef, String> {
        let rf = self.objects.len();

        let mesh = Mesh::new(
            &three_rs::geometry::LineStrip::new(
                lines, color
            )
        );
        
        self.objects.push(mesh.clone());
        self.scene.add(mesh);

        Ok(rf)
    }

    fn add_from_lines(
        &mut self,
        color: RGB,
        lines: Vec<Vector3>
    ) -> Result<ModelRef, String> {
        let rf = self.objects.len();

        let mesh = Mesh::new(
            &three_rs::geometry::Lines::new(
                lines, color
            )
        );
        
        self.objects.push(mesh.clone());
        self.scene.add(mesh);

        Ok(rf)
    }

    fn show_model(
        &mut self,
        rf: ModelRef
    ) {
        self.objects[rf].borrow_mut()
            .show();
    }

    fn hide_model(
        &mut self,
        rf: ModelRef
    ) {
        self.objects[rf].borrow_mut()
            .hide();
    }

    fn set_vertices(
        &mut self,
        rf: ModelRef,
        vertices: Vec<Vector3>
    ) {
        self.objects[rf].borrow_mut()
            .set_positions(vertices);
    }

    fn set_position(
        &mut self,
        rf: ModelRef,
        pos: &Vector3
    ) {
        self.objects[rf].borrow_mut()
            .set_position(pos.clone());
    }

    fn set_rotation(
        &mut self,
        rf: ModelRef,
        q: &Quaternion
    ) {
        self.objects[rf].borrow_mut()
            .set_rotation(q.clone());
    }

    fn set_scale(
        &mut self,
        rf: ModelRef,
        scale: &Vector3
    ) {
        self.objects[rf].borrow_mut()
            .set_scale(scale.clone());
    }

    fn change_state(
        &mut self,
        _model_ref: ModelRef,
        _from: ObjectState,
        _to: ObjectState
    ) {
    }

    fn render(
        &mut self
    ) {
        self.scene.render(
            (&mut self.camera) as &mut dyn ObjectCamera
        );
    }
}