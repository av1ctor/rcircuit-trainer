use std::{cell::RefCell, rc::Rc, fs};
use bot_engine_gltf::renderer::{ModelRef, self};
use three_rs::{
    math::{Vector3, Quaternion, Euler, Triangle}, 
    loader::Gltf, 
    object::Mesh, 
    camera::{PerspectiveCamera, ObjectCamera}, 
    scene::Scene, 
    core::{Transformable, Geometrical, Object3d, RGB}, renderer::Event,
};

pub struct GlRenderer {
    pub w: usize,
    pub h: usize,
    pub objects: Vec<Rc<RefCell<Mesh>>>,
    pub scene: Scene,
    pub camera: PerspectiveCamera,
    pub renderer: Rc<RefCell<three_rs::renderer::GlRenderer>>,
}

impl GlRenderer {
    pub fn new(
        w: u32,
        h: u32
    ) -> Result<Rc<RefCell<Self>>, String> {
        unsafe {
            let renderer = Rc::new(RefCell::new(
                three_rs::renderer::GlRenderer::new("trainer", w, h)
            ));
            let mut camera = PerspectiveCamera::new(75.0, (w as f32) / (h as f32), 0.1, 1000.0);
            let scene = Scene::new(renderer.clone());

            camera.rotate_x(-0.5)
                .translate_y(15.0)
                .translate_z(50.0)
                .translate_x(15.0);

            Ok(Rc::new(RefCell::new(Self{
                w: w as _,
                h: h as _,
                objects: vec![],
                scene,
                camera,
                renderer,
            })))
        }
    }

    pub fn poll_events(
        &mut self
    ) -> Vec<Event> {
        self.renderer.borrow_mut().poll_events()
    }
}

impl renderer::Renderer for GlRenderer {
    fn reset(
        &mut self
    ) {
        self.objects = vec![];
        self.scene.reset();
    }
    
    fn add_from_file(
        &mut self,
        id: &str
    ) -> Result<ModelRef, String> {
        if let Ok(bytes) = fs::read(id) {
            self.add_from_bytes(&bytes)
        }
        else {
            Err("File not found".to_string())
        }
    }

    fn add_from_bytes(
        &mut self,
        bytes: &[u8]
    ) -> Result<ModelRef, String> {
        let mesh = Gltf::load_from_bytes(bytes)?;
        let rf = self.objects.len();

        self.objects.push(mesh.clone());
        self.scene.add(mesh);

        Ok(rf)
    }

    fn add_from_gltf(
        &mut self,
        doc: &gltf::Document,
        buffers: &Vec<gltf::buffer::Data>
    ) -> Result<ModelRef, String> {
        let mesh = Gltf::load(doc, buffers)?;
        let rf = self.objects.len();

        self.objects.push(mesh.clone());
        self.scene.add(mesh);

        Ok(rf)
    }

    fn add_from_mesh(
        &mut self,
        _color: RGB,
        mesh: Rc<RefCell<Mesh>>
    ) -> Result<ModelRef, String> {
        let rf = self.objects.len();

        self.objects.push(mesh.clone());
        self.scene.add(mesh);

        Ok(rf)
    }

    fn add_from_triangles(
        &mut self,
        color: RGB,
        triangles: Vec<Triangle>
    ) -> Result<ModelRef, String> {
        let rf = self.objects.len();

        let mesh = Mesh::new(
            &three_rs::geometry::Triangles::new(
                triangles, color
            )
        );
        
        self.objects.push(mesh.clone());
        self.scene.add(mesh);

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
        self.objects[rf].borrow_mut().show();
    }

    fn hide_model(
        &mut self,
        rf: ModelRef
    ) {
        self.objects[rf].borrow_mut().hide();
    }

    fn set_vertices(
        &mut self,
        rf: ModelRef,
        vertices: Vec<Vector3>
    ) {
        self.objects[rf].borrow_mut().set_positions(vertices);
    }

    fn set_position(
        &mut self,
        rf: ModelRef,
        pos: &Vector3
    ) {
        self.objects[rf].borrow_mut().set_position(pos.clone());
    }

    fn set_rotation(
        &mut self,
        rf: ModelRef,
        q: &Quaternion
    ) {
        self.objects[rf].borrow_mut().set_rotation(q.clone());
    }

    fn set_rotation_from_euler(
        &mut self,
        rf: ModelRef,
        euler: &Euler
    ) {
        self.objects[rf].borrow_mut().set_rotation_from_euler(euler.clone());
    }

    fn set_scale(
        &mut self,
        rf: ModelRef,
        scale: &Vector3
    ) {
        self.objects[rf].borrow_mut().set_scale(scale.clone());
    }

    fn render(
        &mut self
    ) {
        self.scene.render((&mut self.camera) as &mut dyn ObjectCamera);
    }
}