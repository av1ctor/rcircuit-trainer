use std::{cell::RefCell, collections::HashMap, fs, rc::Rc};
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

pub struct ThreeRenderer {
    pub w: usize,
    pub h: usize,
    pub assets: HashMap<AssetRef, Rc<RefCell<Mesh>>>,
    pub objects: Vec<Rc<RefCell<Mesh>>>,
    pub scene: Scene,
    pub camera: PerspectiveCamera,
    pub renderer: Rc<RefCell<three_rs::renderer::GlRenderer>>,
}

impl ThreeRenderer {
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
                assets: HashMap::new(),
            })
        }
    }

    pub fn poll_events(
        &mut self
    ) -> Vec<Event> {
        self.renderer.borrow_mut().poll_events()
    }

    pub fn ticks(
        &mut self
    ) -> u32 {
        self.renderer.borrow_mut().ticks()
    }

    pub fn delay(
        &mut self,
        ms: u32
    ) {
        self.renderer.borrow_mut().delay(ms)
    }
}

impl Renderer for ThreeRenderer {
    fn reset(
        &mut self
    ) {
        self.objects = vec![];
        self.scene.reset();
    }

    fn load_asset(
        &mut self,
        asset_ref: AssetRef,
        asset: &Asset
    ) -> Result<(), String> {
        match GameAssetType::from(asset.typ) {
            GameAssetType::Level => {
                match asset.data.0.get("path").unwrap() {
                    AssetValue::Text(path) => {
                        if let Ok(bytes) = fs::read(path) {
                            let mesh = Gltf::load_from_bytes(&bytes)?;
                            self.assets.insert(asset_ref, mesh);
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
                self.assets.insert(asset_ref, mesh);
    
                Ok(())
            },
            GameAssetType::Npc => {
                let mesh = Mesh::new(
                    &Box3::new(1.0, 1.0, 1.0)
                );
                self.assets.insert(asset_ref, mesh);
    
                Ok(())
            },
            GameAssetType::Exit => {
                let mesh = Mesh::new(
                    &Box3::new(2.0, 2.0, 2.0)
                );
                self.assets.insert(asset_ref, mesh);
    
                Ok(())
            },
        }
    }
    
    fn add_model(
        &mut self,
        asset_ref: AssetRef
    ) -> Result<ModelRef, String> {
        if let Some(mesh) = self.assets.get(&asset_ref) {
            let rf = self.objects.len();
            let cloned = Rc::new(RefCell::new(mesh.borrow().clone()));
            self.objects.push(cloned.clone());
            self.scene.add(cloned);
            Ok(rf)
        }
        else {
            return Err(format!("Asset not found: {}", asset_ref));
        }
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
        //println!("{}: {}->{}", _model_ref, _from, _to);
    }

    fn render(
        &mut self
    ) {
        self.scene.render(
            (&mut self.camera) as &mut dyn ObjectCamera
        );
    }
}