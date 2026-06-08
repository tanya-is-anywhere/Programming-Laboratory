package org.example.controller;
import org.example.spring_lab3_notifications.SpringLab3NotificationsApplication;
import org.example.spring_lab3_notifications.config.SecurityConfig;
import org.example.spring_lab3_notifications.controller.UserController;
import org.example.spring_lab3_notifications.security.JwtAuthenticationFilter;
import org.example.spring_lab3_notifications.service.UserService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.Import;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.boot.test.mock.mockito.MockBean;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;
@SpringBootTest(classes = SpringLab3NotificationsApplication.class)
@Import(SecurityConfig.class)
@AutoConfigureMockMvc
class UserControllerTest {
    @Autowired
    private MockMvc mockMvc;
    @MockBean
    private UserService userService;
    @MockBean
    private JwtAuthenticationFilter jwtAuthenticationFilter;
    @Test
    void shouldReturnOk() throws Exception {
        mockMvc.perform(get("/users/all"))
                .andExpect(status().isOk());
    }
}
