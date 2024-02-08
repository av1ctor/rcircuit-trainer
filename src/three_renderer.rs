use std::{cell::RefCell, collections::HashMap, fs, rc::Rc};
use brains_engine::{
    asset::GameAssetType, 
    consts, 
    entity::EntityState, 
    env::{Asset, AssetRef, AssetValue}, 
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
    pub assets: HashMap<AssetRef, (GameAssetType, Rc<RefCell<Mesh>>)>,
    pub objects: Vec<(AssetRef, Rc<RefCell<Mesh>>)>,
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
                            self.assets.insert(asset_ref, (GameAssetType::Level, mesh));
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
                    &Cone::new(consts::PLAYER_WIDTH, consts::PLAYER_HEIGHT)
                );
                self.assets.insert(asset_ref, (GameAssetType::Player, mesh));
    
                Ok(())
            },
            GameAssetType::Npc => {
                let mesh = Mesh::new(
                    &Box3::new(consts::NPC_WIDTH, consts::NPC_HEIGHT, consts::NPC_WIDTH / 2.0)
                );
                self.assets.insert(asset_ref, (GameAssetType::Npc, mesh));
    
                Ok(())
            },
            GameAssetType::Exit => {
                let mesh = Mesh::new(
                    &Box3::new(consts::EXIT_WIDTH, consts::EXIT_HEIGHT, consts::EXIT_WIDTH / 2.0)
                );
                self.assets.insert(asset_ref, (GameAssetType::Exit, mesh));
    
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
            let cloned = Rc::new(RefCell::new(mesh.1.borrow().clone()));
            self.objects.push((asset_ref, cloned.clone()));
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
        
        self.objects.push((usize::MAX, mesh.clone()));
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
        
        self.objects.push((usize::MAX, mesh.clone()));
        self.scene.add(mesh);

        Ok(rf)
    }

    fn show_model(
        &mut self,
        rf: ModelRef
    ) {
        self.objects[rf].1.borrow_mut()
            .show();
    }

    fn hide_model(
        &mut self,
        rf: ModelRef
    ) {
        self.objects[rf].1.borrow_mut()
            .hide();
    }

    fn set_model_vertices(
        &mut self,
        rf: ModelRef,
        vertices: Vec<Vector3>
    ) {
        self.objects[rf].1.borrow_mut()
            .set_positions(vertices);
    }

    fn set_model_position(
        &mut self,
        rf: ModelRef,
        pos: &Vector3
    ) {
        let obj = &mut self.objects[rf];

        let translate = if obj.0 != usize::MAX {
            let asset = self.assets.get(&obj.0).unwrap();
            match asset.0 {
                GameAssetType::Level => 
                    Vector3::zero(),
                GameAssetType::Player => 
                    Vector3::new(0.0, consts::PLAYER_HEIGHT / 2.0, 0.0),
                GameAssetType::Npc => 
                    Vector3::new(0.0, consts::NPC_HEIGHT / 2.0, 0.0),
                GameAssetType::Exit => 
                    Vector3::new(0.0, consts::EXIT_HEIGHT / 2.0, 0.0),
            }
        }
        else {
            Vector3::zero()
        };

        obj.1.borrow_mut().set_position(pos.add(&translate));
    }

    fn set_model_rotation(
        &mut self,
        rf: ModelRef,
        q: &Quaternion
    ) {
        self.objects[rf].1.borrow_mut()
            .set_rotation(q.clone());
    }

    fn set_model_scale(
        &mut self,
        rf: ModelRef,
        scale: &Vector3
    ) {
        self.objects[rf].1.borrow_mut()
            .set_scale(scale.clone());
    }

    fn change_model_state(
        &mut self,
        _model_ref: ModelRef,
        _from: EntityState,
        _to: EntityState
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