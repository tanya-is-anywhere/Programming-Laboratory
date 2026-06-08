package org.example.service;
import org.example.spring_lab3_notifications.model.dto.UserDto;
import org.example.spring_lab3_notifications.model.entity.User;
import org.example.spring_lab3_notifications.repository.UserRepository;
import org.example.spring_lab3_notifications.service.UserService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
@ExtendWith(MockitoExtension.class)
class UserServiceVerifyTest {
    @Mock
    private UserRepository userRepository;
    @InjectMocks
    private UserService userService;
    @Test
    void shouldCallSaveOnRepository() {
        UserDto dto = UserDto.builder()
                .name("Иван")
                .email("ivan@example.com")
                .build();
        when(userRepository.save(any(User.class))).thenReturn(new User());
        userService.createUser(dto);
        verify(userRepository).save(any(User.class));
    }
}