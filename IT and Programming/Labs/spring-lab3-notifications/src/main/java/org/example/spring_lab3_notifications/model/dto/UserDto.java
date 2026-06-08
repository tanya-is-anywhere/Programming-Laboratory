package org.example.spring_lab3_notifications.model.dto;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import java.time.LocalDateTime;
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserDto {
    @NotBlank(message = "Имя не должно быть пустым")
    @Size(max = 100, message = "Имя не должно быть длиннее 100 символов")
    private String name;
    @NotBlank(message = "Email обязателен")
    @Email(message = "Некорректный формат email")
    @Pattern(
            regexp = "^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+$",
            message = "Email не соответствует требуемому шаблону"
    )
    private String email;
    @Pattern(
            regexp = "^\\+?\\d{1,3} \\d{3} \\d{3}-\\d{2}-\\d{2}$",
            message = "Email не соответствует требуемому шаблону"
    )
    private String phone;
    private String deviceToken;
    private String telegramChatId;
    private LocalDateTime createdAt;

}