use std::f32::consts::PI;

use bot_engine_gltf::renderer::{Renderer, ModelRef};
use glow::{HasContext, NativeProgram, NativeBuffer, NativeVertexArray};
use three_rs::math::{vector3::Vector3, matrix4::Matrix4};

const VERTEX_SHADER_SOURCE: &str = include_str!("../shaders/vertex.glsl");
const FRAGMENT_SHADER_SOURCE: &str = include_str!("../shaders/frag.glsl");

pub struct GlRenderer {
    pub gl: glow::Context,
    pub window: sdl2::video::Window,
    pub gl_context: sdl2::video::GLContext,
    pub event_loop: sdl2::EventPump,
    pub program: NativeProgram,
    pub vbo: NativeBuffer,
    pub vao: NativeVertexArray,

    pub model_mat: Matrix4,
    pub model_rot: Matrix4,
    pub model_pos: Vector3,
    pub view_mat: Matrix4,
}

impl Drop for GlRenderer {
    fn drop(
        &mut self
    ) {
        unsafe {
            self.gl.delete_program(self.program);
            self.gl.delete_vertex_array(self.vao);
            self.gl.delete_buffer(self.vbo)
        }
    }
}

impl GlRenderer {
    pub fn new(
        w: u32,
        h: u32
    ) -> Result<Self, String> {
        let sdl = sdl2::init().unwrap();
        let video = sdl.video().unwrap();
        let gl_attr = video.gl_attr();
        gl_attr.set_context_profile(sdl2::video::GLProfile::Core);
        gl_attr.set_context_version(3, 3);
        let window = video
            .window("trainer", w, h)
            .opengl()
            .resizable()
            .build()
            .unwrap();
        let gl_context = window.gl_create_context().unwrap();
        let gl = unsafe {
            glow::Context::from_loader_function(|s| video.gl_get_proc_address(s) as *const _)
        };

        let event_loop = sdl.event_pump().unwrap();
        
        unsafe {
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LESS);

            let program = Self::create_program(&gl, VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE);
            gl.use_program(Some(program));

            // Create a vertex buffer and vertex array object
            let (vbo, vao) = Self::create_vertex_buffer(&gl);

            // Model matrix : an identity matrix (model will be at the origin)
            let model_mat = Matrix4::identity();
            let model_rot = Matrix4::identity();
            let model_pos = Vector3::new(-3.0, 0.0, -3.0);

            // generate a camera view, based on eye-position and lookAt world-position
            let view_mat = Self::gen_view(&Vector3::new(2.0, 2.0, 20.0), &model_pos);
            
            let sun_position = Vector3::new(3.0, 10.0, -5.0);
            let sun_color = Vector3::one();

            Self::set_uniform_3fv(&gl, program, "sun_position", &sun_position.to_slice());
            Self::set_uniform_3fv(&gl, program, "sun_color", &sun_color.to_slice());

            gl.clear_color(0.1, 0.2, 0.3, 1.0);
            
            Ok(Self{
                gl, 
                window, 
                event_loop, 
                gl_context,
                program,
                vbo,
                vao,
                model_mat,
                model_rot,
                model_pos,
                view_mat,
            })
        }
        
    }

    unsafe fn create_program(
        gl: &glow::Context,
        vert_shader_src: &str,
        frag_shader_src: &str
    ) -> NativeProgram {
        let program = gl.create_program().expect("Cannot create program");

        let shader_sources = [
            (glow::VERTEX_SHADER, vert_shader_src),
            (glow::FRAGMENT_SHADER, frag_shader_src),
        ];
    
        let mut shaders = Vec::with_capacity(shader_sources.len());
    
        for (shader_type, shader_source) in shader_sources.iter() {
            let shader = gl
                .create_shader(*shader_type)
                .expect("Cannot create shader");
            gl.shader_source(shader, shader_source);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                panic!("{}", gl.get_shader_info_log(shader));
            }
            gl.attach_shader(program, shader);
            shaders.push(shader);
        }
    
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            panic!("{}", gl.get_program_info_log(program));
        }
    
        for shader in shaders {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }
    
        program
    }

    unsafe fn create_vertex_buffer(
        gl: &glow::Context
    ) -> (NativeBuffer, NativeVertexArray) {
        // This is a flat array of f32s that are to be interpreted as vec2s.
        let triangle_vertices = [0.5f32, 1.0f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32];
        let triangle_vertices_u8: &[u8] = core::slice::from_raw_parts(
            triangle_vertices.as_ptr() as *const u8,
            triangle_vertices.len() * core::mem::size_of::<f32>(),
        );
    
        // We construct a buffer and upload the data
        let vbo = gl.create_buffer().unwrap();
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, triangle_vertices_u8, glow::STATIC_DRAW);
    
        // We now construct a vertex array to describe the format of the input buffer
        let vao = gl.create_vertex_array().unwrap();
        gl.bind_vertex_array(Some(vao));
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 8, 0);
    
        (vbo, vao)
    }
    
    unsafe fn set_uniform_3fv(
        gl: &glow::Context, 
        program: NativeProgram, 
        name: &str, 
        value: &[f32; 3]
    ) {
        let uniform_location = gl.get_uniform_location(program, name);
        gl.uniform_1_f32_slice(uniform_location.as_ref(), value)
    }

    unsafe fn set_uniform_matrix_4fv(
        gl: &glow::Context, 
        program: NativeProgram, 
        name: &str, 
        value: &[f32; 4*4]
    ) {
        let uniform_location = gl.get_uniform_location(program, name);
        gl.uniform_matrix_4_f32_slice(uniform_location.as_ref(), false, value);
    }

    fn gen_view(
        pos: &Vector3, 
        lookat: &Vector3
    ) -> Matrix4 {
        // Camera matrix
        let view = Matrix4::look_at(
            pos,                // Camera in World Space
            lookat,             // and looks at the origin
            &Vector3::new(0.0, 1.0, 0.0)  // Head is up (set to 0,-1,0 to look upside-down)
        );
      
        view
    }

    fn gen_modelview_projection(
        view_mat: &Matrix4, 
        model_mat: &Matrix4, 
        fov: f32, 
        w: usize,
        h: usize
    ) -> Matrix4 {
        let projection = Matrix4::perspective_fov(fov, w as f32 / h as f32, 0.01, 1000.0, 1.0);

        // In world coordinates
        projection.mul(view_mat).mul(model_mat)
    }

    pub fn render(
        &mut self
    ) {
        let (w, h) = self.window.size();
        
        unsafe {
            let trans = Matrix4::from_translation(&self.model_pos);  // reposition model
            self.model_rot = self.model_rot.rotate(0.8 * PI / 180.0, &Vector3::new(0.0, 1.0, 0.0));  // rotate model on y axis
            self.model_mat = trans.mul(&self.model_rot);
            
            let mvp = Self::gen_modelview_projection(&self.view_mat, &self.model_mat, 45.0, w as _, h as _);
            Self::set_uniform_matrix_4fv(&self.gl, self.program, "MVP", &mvp.0);

            self.gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
            self.gl.draw_arrays(glow::TRIANGLES, 0, 3);
        }
        
        self.window.gl_swap_window();
    }
}

impl Renderer for GlRenderer {
    fn load_model(
        &mut self,
        _name: &str
    ) -> ModelRef {
        todo!()
    }

    fn add_model(
        &mut self,
        _model_ref: ModelRef
    ) {
        todo!()
    }

    fn show_model(
        &mut self,
        _model_ref: ModelRef
    ) {
        todo!()
    }

    fn hide_model(
        &mut self,
        _model_ref: ModelRef
    ) {
        todo!()
    }

    fn set_position(
        &mut self,
        _model_ref: ModelRef,
        _pos: &Vector3
    ) {
        todo!()
    }

    fn set_rotation(
        &mut self,
        _model_ref: ModelRef,
        _rot: &Vector3,
        _q: &three_rs::math::quaternion::Quaternion
    ) {
        todo!()
    }

    fn set_scale(
        &mut self,
        _model_ref: ModelRef,
        _scale: &Vector3
    ) {
        todo!()
    }
}